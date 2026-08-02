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
    The `cpu` extra installs the CPU-only build of PyTorch. The `cuda` extra installs PyTorch from the cu128 wheel index:
    ```shell
    pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    ```
    The GPU Docker image is based on CUDA 13 (`nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04`). The cu128 PyTorch wheels bundle their own CUDA 12.8 runtime libraries and run fine inside the CUDA 13 image; what matters is the host NVIDIA driver version (use a driver that supports CUDA 13, i.e. >= 580, when running the GPU image). PyTorch 2.8 has no cu130 wheels, which is why the wheel index stays cu128 for now.

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