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

Available ASR_MODELs are `tiny`, `base`, `small`, `medium`, `large` (only OpenAI Whisper), `large-v1`, `large-v2` and `large-v3`.

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.


### Configuring the `Model Path`

```sh
export ASR_MODEL_PATH=/data/whisper
```

### Configuring the `ALLOW_ORIGINS`

```sh
export ALLOW_ORIGINS
```
**Default: `*`**

List the origins that are allowed to access the model. Wild cards are supported.

**Docker-compose example**
```
services:
  openai-whisper-asr:
    image: onerahmet/openai-whisper-asr-webservice:latest
    environment:
      - ASR_MODEL=base
      - ASR_ENGINE=openai_whisper
      - ALLOW_ORIGINS=https://example.com
```
**Adding multiple**
```
- ALLOW_ORIGINS=https://example.com, https://second_example.com, https://third_example.com
```
**Wildcards**
```
- ALLOW_ORIGINS=https://*.exmaple.com
```