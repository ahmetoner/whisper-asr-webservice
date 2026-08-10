# M4A Fix for whisper-asr-webservice

**TL;DR:** M4A files fail because ffmpeg can't seek when audio is piped via stdin; fix writes M4A files to temp disk first.

Building the container also fails,.

---

## Solution

Patch the upstream GPU image with fixed Python files:

```dockerfile
# Dockerfile.patch
FROM onerahmet/openai-whisper-asr-webservice:latest-gpu
COPY app/utils.py /app/app/utils.py
COPY app/webservice.py /app/app/webservice.py
```

Build and run:
```bash
docker build -f Dockerfile.patch -t oneit/whisper-asr:gpu-m4a-fix .
docker push oneit/whisper-asr:gpu-m4a-fix

docker run -d --gpus all -p 9000:9000 \
  -e ASR_MODEL=small.en \
  -v /var/cache/whisper:/root/.cache/whisper \
  oneit/whisper-asr:gpu-m4a-fix
```

---

## The Problem

### Symptoms
- Some M4A files transcribe fine, others silently fail (empty response)
- MP3 files always work
- Same M4A file works if converted to MP3 first

### Root Cause

M4A files use the MP4 container format which has a "moov atom" containing metadata (duration, codec info, etc.). This atom can be located either:

1. **At the start** (faststarted) - works with streaming
2. **At the end** (default) - requires seeking, breaks stdin piping

The whisper-asr-webservice pipes uploaded audio directly to ffmpeg via stdin:

```python
# Original code in app/utils.py
ffmpeg.input("pipe:", threads=0)  # stdin - no seeking possible
    .run(input=file.read())
```

When ffmpeg receives an M4A file via stdin with the moov atom at the end, it cannot seek backward to read the metadata, causing decode failure.

### Why Some M4A Files Work

Files recorded on certain devices (e.g., some Android recorders) are "faststarted" - the moov atom is at the beginning. These work fine via stdin. Files from other sources (iOS voice memos, many audio editors) have moov at the end and fail.

---

## The Fix

**Files changed:** `app/utils.py`, `app/webservice.py` (~20 lines total)

The fix detects M4A/MP4/MOV files by extension and writes them to a temporary file before processing, allowing ffmpeg to seek:

```python
# New code in app/utils.py
if filename and filename.lower().endswith(('.m4a', '.mp4', '.mov', '.m4v')):
    # Write to temp file to allow ffmpeg to seek for moov atom
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    try:
        ffmpeg.input(tmp_path, threads=0)  # file path - seeking works
            .run()
    finally:
        os.unlink(tmp_path)
else:
    # Original stdin pipe for other formats (MP3, WAV, etc.)
    ffmpeg.input("pipe:", threads=0)
        .run(input=file.read())
```

### Why Not Fix ffmpeg-container?

Initial investigation suggested the issue was missing AAC codecs in `ahmetoner/ffmpeg-container`. Testing proved this wrong - the native AAC decoder is included. The real issue is the stdin streaming limitation with non-faststarted M4A files.

### Comparison to PR #294

PR #294 adds pydub and recompiles ffmpeg from source (+133/-15 lines, Dockerfile changes). This fix is minimal Python-only changes (~20 lines) that patch on top of the existing upstream image.

---

## Testing

```
File                      moov position    Before    After
---------------------------------------------------------
test01-audio.m4a          at start         PASS      PASS
test02-audio.m4a          at end           FAIL      PASS
test03-audio.m4a          at end           FAIL      PASS
test04-audio.m4a          at end           FAIL      PASS
test05-audio.m4a          at end           FAIL      PASS
test02-audio.mp3          n/a              PASS      PASS
```

---

## Links

- PR submitted: https://github.com/ahmetoner/whisper-asr-webservice/pull/XXX
- Related issue: https://github.com/ahmetoner/whisper-asr-webservice/issues/272
- Alternative PR (heavier approach): https://github.com/ahmetoner/whisper-asr-webservice/pull/294
