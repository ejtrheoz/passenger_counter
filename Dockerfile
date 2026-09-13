FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# BPJDet person detector source (the .pt checkpoint unpickles classes from it).
ARG BPJDET_COMMIT=06821caaeb3daa7fc28867abd32685135ad2cbaf
RUN git init -q third_party/BPJDet && cd third_party/BPJDet \
    && git fetch -q --depth 1 https://github.com/hnuzhy/BPJDet.git ${BPJDET_COMMIT} \
    && git checkout -q FETCH_HEAD \
    && rm -rf .git test_imgs materials demos data \
    # silence per-image "no object detected" print in post_process_batch
    && sed -i 's/print("This image has no object detected!")/pass/' val.py

# CUDA 13.0 wheels bundle the CUDA runtime; the host only needs the NVIDIA driver + container toolkit.
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 torchvision==0.26.0+cu130

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY door-flow ./door-flow
COPY ksiva ./ksiva
COPY app ./app

ENV ASSETS_DIR=/app/assets \
    WORK_DIR=/data/work \
    DEVICE=0 \
    PORT=80

# Runpod load-balancing endpoints inject PORT (default 80) and probe /ping.
EXPOSE 80
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
