## Development Environment

Install poetry with following command:

```sh
pip3 install poetry
```

Install torch with following command:

```sh
# just for GPU:
pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch
```

### Run

Install packages:

```sh
poetry install
```

Starting the Webservice:

```sh
poetry run gunicorn --bind 0.0.0.0:9000 --workers 1 --timeout 0 app.webservice:app -k uvicorn.workers.UvicornWorker
```

### Build

=== ":octicons-file-code-16: `Poetry`"

    Build .whl package
    
    ```sh
    poetry build
    ```
=== ":octicons-file-code-16: `Docker`"

    With `Dockerfile`:

    === ":octicons-file-code-16: `CPU`"
    
        ```sh
        # Build Image
        docker build -t whisper-asr-webservice .
        
        # Run Container
        docker run -d -p 9000:9000 whisper-asr-webservice
        # or
        docker run -d -p 9001:9000 -e ASR_MODEL=base whisper-asr-webservice3
        ```
    
    === ":octicons-file-code-16: `GPU`"
    
        ```sh
        # Build Image
        docker build -f Dockerfile.gpu -t whisper-asr-webservice-gpu .
        
        # Run Container
        docker run -d --gpus all -p 9000:9000 whisper-asr-webservice-gpu
        # or
        docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice-gpu
        ```

    With `docker-compose`:
    
    === ":octicons-file-code-16: `CPU`"
    
        ```sh
        docker-compose up --build
        ```
    
    === ":octicons-file-code-16: `GPU`"
    
        ```sh
        docker-compose up --build -f docker-compose.gpu.yml
        ```

