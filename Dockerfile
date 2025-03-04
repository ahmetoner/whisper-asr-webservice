FROM debian:bookworm-slim AS ffmpeg

ARG FFMPEG_VERSION=n7.1

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    yasm \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FFmpeg/FFmpeg.git --depth 1 --branch $FFMPEG_VERSION --single-branch /FFmpeg

WORKDIR /FFmpeg

RUN PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
      --prefix="$HOME/ffmpeg_build" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I$HOME/ffmpeg_build/include" \
      --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
      --extra-libs="-lpthread -lm" \
      --ld="g++" \
      --bindir="$HOME/bin" \
      --disable-doc \
      --disable-htmlpages \
      --disable-podpages \
      --disable-txtpages \
      --disable-network \
      --disable-autodetect \
      --disable-hwaccels \
      --enable-ffprobe \
      --disable-ffplay \
      --enable-filter=copy \
      --enable-protocol=file \
      --enable-small && \
    PATH="$HOME/bin:$PATH" make -j$(nproc) && \
    make install && \
    hash -r

FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

FROM python:3.10-bookworm

ENV POETRY_VENV=/app/.venv

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.1.1

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY . /app
COPY --from=ffmpeg /root/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /root/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

RUN poetry config virtualenvs.in-project true
RUN poetry install

EXPOSE 9000

ENTRYPOINT ["whisper-asr-webservice"]
