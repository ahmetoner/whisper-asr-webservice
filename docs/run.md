## Usage

Whisper ASR Webservice now available on Docker Hub. You can find the latest version of this repository on docker hub for CPU and GPU.

Docker Hub: <https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice>

=== ":octicons-file-code-16: `CPU`"

    ```shell
    docker pull onerahmet/openai-whisper-asr-webservice:latest
    docker run -d -p 9000:9000 \
      -e ASR_MODEL=base \
      -e ASR_ENGINE=openai_whisper \
      onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `CPU (macOS)`"

    > GPU passthrough does not work on macOS due to fundamental design limitations of Docker. Docker actually runs containers within a LinuxVM on macOS. If you wish to run GPU-accelerated containers, I'm afraid Linux is your only option.
    > 
    > The `:latest` image tag provides both amd64 and arm64 architectures:
    
    ```shell
    docker pull onerahmet/openai-whisper-asr-webservice:latest
    docker run -d -p 9000:9000 \
      -e ASR_MODEL=base \
      -e ASR_ENGINE=openai_whisper \
      onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `GPU`"

    ```shell
    docker pull onerahmet/openai-whisper-asr-webservice:latest-gpu
    docker run -d --gpus all -p 9000:9000 \
      -e ASR_MODEL=base \
      -e ASR_ENGINE=openai_whisper \
      onerahmet/openai-whisper-asr-webservice:latest-gpu
    ```

### Environment Variables

The following environment variables can be used to configure the service:

- `ASR_MODEL`: Whisper model to use (tiny, base, small, medium, large) [default: base]
- `ASR_ENGINE`: ASR engine to use (openai_whisper, faster_whisper) [default: openai_whisper]
- `ASR_MODEL_PATH`: Custom path to store/load model files [optional]

> Interactive Swagger API documentation is available at <http://localhost:9000/docs>

![Swagger UI](assets/images/swagger-ui.png)

## Cache

The ASR model is downloaded each time you start the container. Using the large model can take significant time to download.
To reduce container startup time by avoiding repeated downloads, you can persist the cache directory to local storage.
The model will then be loaded from the cache instead of being downloaded again on subsequent container starts.

**Important: Using a persistent cache will prevent you from receiving model updates.**

=== ":octicons-file-code-16: `Default cache dir`"

    ```shell
    docker run -d -p 9000:9000 \
      -v $PWD/cache:/root/.cache \
      onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `With ASR_MODEL_PATH`"

    ```shell
    docker run -d -p 9000:9000 \
      -e ASR_MODEL_PATH=/data/whisper \
      -v $PWD/cache:/data/whisper \
      onerahmet/openai-whisper-asr-webservice:latest
    ```
