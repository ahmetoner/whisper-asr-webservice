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

Available ASR_MODELs are `tiny`, `base`, `small`, `medium`, `large`, `large-v1`, `large-v2`, `large-v3`, `turbo`(only OpenAI Whisper) and `large-v3-turbo`(only OpenAI Whisper).

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

### Configuring the `Model Path`

```sh
export ASR_MODEL_PATH=/data/whisper
```
