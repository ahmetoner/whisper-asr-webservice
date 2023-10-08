## Usage

Whisper ASR Webservice now available on Docker Hub. You can find the latest version of this repository on docker hub for CPU and GPU.

Docker Hub: <https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice>

=== ":octicons-file-code-16: `CPU`"

    ```sh
    docker pull onerahmet/openai-whisper-asr-webservice:latest
    docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `CPU (macOS)`"

    > GPU passthrough does not work on macOS due to fundamental design limitations of Docker. Docker actually runs containers within a LinuxVM on macOS. If you wish to run GPU-accelerated containers, I'm afraid Linux is your only option.
    > 
    > The `:latest` image tag provides both amd64 and arm64 architectures:
    
    ```sh
    docker pull onerahmet/openai-whisper-asr-webservice:latest
    docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `GPU`"

    ```sh
    docker pull onerahmet/openai-whisper-asr-webservice:latest-gpu
    docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
    ```

> Interactive Swagger API documentation is available at http://localhost:9000/docs

![Swagger UI](assets/images/swagger-ui.png)

## Cache
The ASR model is downloaded each time you start the container, using the large model this can take some time. 
If you want to decrease the time it takes to start your container by skipping the download, you can store the cache directory (`~/.cache/whisper` or `/root/.cache/whisper`) to a persistent storage. 
Next time you start your container the ASR Model will be taken from the cache instead of being downloaded again.

**Important this will prevent you from receiving any updates to the models.**

=== ":octicons-file-code-16: `Default cache dir`"

    ```sh
    docker run -d -p 9000:9000 -v $PWD/yourlocaldir:/root/.cache/whisper onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `With ASR_MODEL_PATH`"

    ```sh
    docker run -d -p 9000:9000 -e ASR_MODEL_PATH=/data/whisper -v $PWD/yourlocaldir:/data/whisper onerahmet/openai-whisper-asr-webservice:latest
    ```
