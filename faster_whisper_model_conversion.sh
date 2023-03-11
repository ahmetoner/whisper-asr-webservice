#!/bin/bash

ct2-transformers-converter --model openai/whisper-"$ASR_MODEL" --output_dir /root/.cache/faster_whisper --quantization float16
