WORKDIR := /tmp/whisper_asr_webservice
DOCKER_IMAGE_LATEST := whisper-asr-webservice:latest

dev-up:
	 poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
clean:
	rm -rf $(WORKDIR)

build-cpu-image:
	docker buildx build . -t $(DOCKER_IMAGE_LATEST) -f Dockerfile

build-cpu-image-arm:
	docker buildx build . --platform linux/arm64 -t $(DOCKER_IMAGE_LATEST) -f Dockerfile

build-gpu-image:
	docker buildx build . -t $(DOCKER_IMAGE_LATEST)-gpu -f Dockerfile.gpu
