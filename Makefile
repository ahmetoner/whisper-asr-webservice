WORKDIR := /tmp/whisper_asr_webservice

dev-up:
	 poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
clean:
	rm -rf $(WORKDIR)