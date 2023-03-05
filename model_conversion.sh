#!/bin/bash

# TODO: Add step to build setup based on ENV variable
ct2-transformers-converter --model openai/whisper-large-v2 --output_dir /root/.cache/faster_whisper --quantization float16
