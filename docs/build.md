## Development Environment

Install poetry v2.X with following command:

```shell
pip3 install poetry
```

### Installation

Install dependencies for cpu

```shell
poetry install --extras cpu
```

Install dependencies for cuda

```shell
poetry install --extras cuda
```

!!! Note
    Python 3.12 or 3.13 is required. The `cpu` extra installs the CPU build of PyTorch, and the `cuda` extra
    installs PyTorch 2.8 built for CUDA 12.8 (supports GPUs up to and including the RTX 50xx / Blackwell series).
    You need an NVIDIA driver that supports CUDA 12.8 or newer.

### Run

Starting the Webservice:

```shell
poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
```

### Build

=== ":octicons-file-code-16: `Docker`"

    With `Dockerfile`:

    === ":octicons-file-code-16: `CPU`"
    
        ```shell
        # Build Image
        docker build -t whisper-asr-webservice .
        
        # Run Container
        docker run -d -p 9000:9000 whisper-asr-webservice
        # or with specific model
        docker run -d -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice
        ```
    
    === ":octicons-file-code-16: `GPU`"
    
        ```shell
        # Build Image
        docker build -f Dockerfile.gpu -t whisper-asr-webservice-gpu .
        
        # Run Container
        docker run -d --gpus all -p 9000:9000 whisper-asr-webservice-gpu
        # or with specific model
        docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice-gpu
        ```

    With `docker-compose`:
    
    === ":octicons-file-code-16: `CPU`"
    
        ```shell
        docker-compose up --build
        ```
    
    === ":octicons-file-code-16: `GPU`"
    
        ```shell
        docker-compose -f docker-compose.gpu.yml up --build
        ```
=== ":octicons-file-code-16: `Poetry`"

    Build .whl package
    
    ```shell
    poetry build
    ```