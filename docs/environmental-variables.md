### Configuring the `Engine`

=== ":octicons-file-code-16: `openai_whisper`"
```sh
export ASR_ENGINE=openai_whisper
```
=== ":octicons-file-code-16: `faster_whisper`"
```sh
export ASR_ENGINE=faster_whisper
```

### Configuring the `Model`

```sh
export ASR_MODEL=base
```

Available ASR_MODELs are `tiny`, `base`, `small`, `medium`, `large`, `large-v1`, `large-v2`, `large-v3`, `turbo` and `large-v3-turbo`.

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en`
models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

### Configuring the `Model Path`

```sh
export ASR_MODEL_PATH=/data/whisper
```

### Configuring the `Model Unloading Timeout`

```sh
export MODEL_IDLE_TIMEOUT=300
```

Defaults to `0`. After no activity for this period (in seconds), unload the model until it is requested again. Setting
`0` disables the timeout, keeping the model loaded indefinitely.

### Configuring the `SAMPLE_RATE`

```sh
export SAMPLE_RATE=16000
```

Defaults to `16000`. Default sample rate for audio input. `16 kHz` is commonly used in `speech-to-text` tasks.


