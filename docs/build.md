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

Install dependencies for intel xpu

```shell
poetry install --extras xpu
```

!!! Note
    By default, this will install the CPU version of PyTorch. For GPU support, you'll need to install the appropriate CUDA version of PyTorch separately:
    ```shell
    # For CUDA support (example for CUDA 11.8):
    pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
    ```

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
    
    === ":octicons-file-code-16: `GPU (cuda)`"
    
        ```shell
        # Build Image
        docker build -f Dockerfile.cuda -t whisper-asr-webservice-cuda .
        
        # Run Container
        docker run -d --gpus all -p 9000:9000 whisper-asr-webservice-cuda
        # or with specific model
        docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice-cuda
        ```

    === ":octicons-file-code-16: `GPU (intel)`"
    
        ```shell
        # Build Image
        docker build -f Dockerfile.intel -t whisper-asr-webservice-intel .
        
        # Run Container
        docker run -d --device=/dev/dri all -p 9000:9000 whisper-asr-webservice-intel
        # or with specific model
        docker run -d --device=/dev/dri all -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice-intel
        ```

    With `docker-compose`:
    
    === ":octicons-file-code-16: `CPU`"
    
        ```shell
        docker-compose up --build
        ```
    
    === ":octicons-file-code-16: `GPU (cuda)`"
    
        ```shell
        docker-compose -f docker-compose.cuda.yml up --build
        ```
    === ":octicons-file-code-16: `GPU (intel)`"
    
        ```shell
        docker-compose -f docker-compose.intel.yml up --build
        ```
=== ":octicons-file-code-16: `Poetry`"

    Build .whl package
    
    ```shell
    poetry build
    ```