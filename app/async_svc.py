import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Literal
from uuid import uuid4

from fastapi import UploadFile
from pydantic import BaseModel

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModel
from app.utils import load_audio

JobStatus = Literal["pending", "processing", "completed", "read", "failed", "error"]


class NewAsyncJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    place_in_queue: int | None = None


class AsyncJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float | None = None
    completed_at: float | None = None
    results: List[Dict] | None = None
    error: str | None = None


class AsyncProcessing:
    def __init__(self, asr_model: ASRModel):
        self.workdir = Path(CONFIG.WORKDIR)
        self.db_path = self.workdir / "data.db"
        self.temp_store = self.workdir / "files"

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.temp_store.mkdir(parents=True, exist_ok=True)
        self.init_db()
        self.db_lock = threading.Lock()
        self.process_lock = threading.Lock()

        self.asr_model = asr_model

        # Start background processing thread
        self.processing_thread = threading.Thread(name="async_processing", target=self._process_jobs, daemon=True)
        self.cleanup_thread = threading.Thread(name="cleanup_old_jobs", target=self.cleanup_old_jobs, daemon=True)

        # Start background threads
        self.processing_thread.start()
        self.cleanup_thread.start()

    def init_db(self):
        """Setup the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    file_location TEXT,
                    results_file TEXT,
                    error TEXT,
                    params TEXT
                )
            """)
            conn.commit()

    def cleanup_old_jobs(self):
        """Remove old jobs from the database and filesystem"""
        abandoned_time_max = CONFIG.JOB_CLEANUP_ABANDONED
        read_time_max = CONFIG.JOB_CLEANUP_AFTER_READ

        while True:
            cutoff_time = time.time() - read_time_max
            abandoned_cutoff_time = time.time() - abandoned_time_max
            try:
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as conn:
                        # Get old job files to delete
                        cursor_1 = conn.execute(
                            "SELECT id, file_location, results_file FROM jobs WHERE created_at < ? AND (status = 'read' OR status = 'failed')",
                            (cutoff_time,),
                        )
                        cursor_2 = conn.execute(
                            "SELECT id, file_location, results_file FROM jobs WHERE created_at < ? AND (status = 'pending' OR status = 'processing')",
                            (abandoned_cutoff_time,),
                        )
                        old_jobs = cursor_1.fetchall() + cursor_2.fetchall()

                        # Delete files
                        for _, file_location, results_file in old_jobs:
                            if file_location:
                                file_path = Path(file_location)
                                if file_path.exists() and file_path.is_dir():
                                    shutil.rmtree(file_path)
                            if results_file:
                                results_path = Path(results_file)
                                if results_path.exists():
                                    results_path.unlink(missing_ok=True)

                        # Delete database records
                        job_ids = [job[0] for job in old_jobs]
                        if job_ids:
                            placeholders = ",".join(["?" for _ in job_ids])
                            conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", job_ids)
                            conn.commit()

                time.sleep(60)
            except Exception as e:
                print(f"Error in cleanup_old_jobs: {e}")
                time.sleep(5)

    async def create_job(self, files: List[UploadFile], params: Dict) -> NewAsyncJobResponse:
        """Create a new job and store files to temp_store"""
        job_id = str(uuid4())
        job_dir = self.temp_store / job_id
        job_dir.mkdir(exist_ok=True)

        # Store uploaded files
        stored_files = []
        for _, file in enumerate(files):
            file_path = job_dir / f"{file.filename}"
            with open(file_path, "wb") as f:
                content = file.file.read()
                f.write(content)
            stored_files.append(str(file_path))

        # Create job record in database and get queue position
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                created_at = time.time()
                conn.execute(
                    "INSERT INTO jobs (id, status, created_at, file_location, params) VALUES (?, ?, ?, ?, ?)",
                    (job_id, "pending", created_at, str(job_dir), json.dumps(params)),
                )

                # Get the position of this job in the queue
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'pending' AND (created_at < ? OR (created_at = ? AND id <= ?))",
                    (created_at, created_at, job_id),
                )
                place_in_queue = cursor.fetchone()[0]

                conn.commit()

        return NewAsyncJobResponse(job_id=job_id, status="pending", place_in_queue=place_in_queue)

    async def get_job(self, job_id: str) -> AsyncJobResponse:
        """Get job status and results"""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id, status, created_at, completed_at, file_location, results_file, error FROM jobs WHERE id = ?",
                    (job_id,),
                )
                row = cursor.fetchone()

        if not row:
            return AsyncJobResponse(job_id=job_id, status="error", error="Job not found. It may have expired.")

        job_id, status, created_at, completed_at, file_location, results_file, error = row

        job_data = AsyncJobResponse(
            job_id=job_id,
            status=status,
            created_at=created_at,
            completed_at=completed_at,
            results=None,
            error=error,
        )
        complete_states = ["completed", "read"]
        if status in complete_states and results_file:
            try:
                with open(results_file, "r") as f:
                    job_data.results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                job_data.error = "Results file not found or corrupted"
        elif status == "failed" and error:
            job_data.error = error

        if status == "completed":
            # Mark job as read
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("UPDATE jobs SET status = 'read' WHERE id = ?", (job_id,))
                    conn.commit()

        return job_data

    def _process_jobs(self):
        """Background thread to process pending jobs"""
        while True:
            try:
                # Get next pending job
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            'SELECT id, file_location, params FROM jobs WHERE status = "pending" ORDER BY created_at LIMIT 1'
                        )
                        row = cursor.fetchone()

                        if row:
                            job_id, file_location, params_json = row
                            # Mark as processing
                            conn.execute('UPDATE jobs SET status = "processing" WHERE id = ?', (job_id,))
                            conn.commit()

                if row:
                    self._process_single_job(job_id, file_location, json.loads(params_json))
                else:
                    # No pending jobs, sleep
                    time.sleep(1)

            except Exception as e:
                print(f"Error in job processing thread: {e}")
                time.sleep(5)

    def _process_single_job(self, job_id: str, file_path: str, params: Dict):
        """Process a single job"""
        with self.process_lock:
            try:
                results = []
                path = Path(file_path)
                for file_name in os.listdir(file_path):
                    with open(path / file_name, "rb") as f:
                        result_stream = self.asr_model.transcribe(
                            audio=load_audio(f, params.get("encode", True)),
                            task=params.get("task", "transcribe"),
                            language=params.get("language"),
                            initial_prompt=params.get("initial_prompt"),
                            vad_filter=params.get("vad_filter", False),
                            word_timestamps=params.get("word_timestamps", False),
                            options={
                                "diarize": params.get("diarize", False),
                                "min_speakers": params.get("min_speakers"),
                                "max_speakers": params.get("max_speakers"),
                            },
                            output=params.get("output", "txt"),
                        )

                        # Read the entire stream result
                        if result_stream:
                            content = result_stream.read()
                        else:
                            content = ""

                        results.append({"filename": file_name, "content": content})

                # Save results to file
                results_file = self.temp_store / job_id / "results.json"
                with open(results_file, "w") as f:
                    json.dump(results, f)

                # Update job status
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            'UPDATE jobs SET status = "completed", completed_at = ?, results_file = ? WHERE id = ?',
                            (time.time(), str(results_file), job_id),
                        )
                        conn.commit()

            except Exception as e:
                # Mark job as failed
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as conn:
                        print(f"Marking job {job_id} as failed: {e}")
                        conn.execute(
                            'UPDATE jobs SET status = "failed", completed_at = ?, error = ? WHERE id = ?',
                            (time.time(), str(e), job_id),
                        )
                        conn.commit()
