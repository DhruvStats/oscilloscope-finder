# Oscilloscope finder - hosted demo image (CPU only)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch keeps the image and the memory footprint small
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# YOLOX source pinned to release 0.3.0 (the PyPI package does not install on Python 3.12)
RUN git clone --depth 1 --branch 0.3.0 https://github.com/Megvii-BaseDetection/YOLOX.git models/yolox \
    && test "$(git -C models/yolox rev-parse HEAD)" = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3" \
    && rm -rf models/yolox/.git models/yolox/demo models/yolox/docs models/yolox/assets

# COCO "everyday objects" model (bottle, chair, laptop ...): labels everything else in the photo.
# Build with --build-arg WITH_CONTEXT=0 to leave it out (and set CONTEXT_MODEL=off).
ARG WITH_CONTEXT=1
RUN if [ "$WITH_CONTEXT" = "1" ]; then \
      curl -fL -o models/yolox_tiny.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth \
      && echo "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0  models/yolox_tiny.pth" | sha256sum -c - ; \
    fi

COPY server ./server
COPY config/instruments.yaml config/registry.py config/__init__.py ./config/
COPY training/yolox_tiny_osc3.py training/yolox_tiny_rtb2004.py ./training/
COPY web/index.html ./web/index.html
COPY models/deploy ./models/deploy

ENV TARGET_EXP=yolox_tiny_osc3 \
    TARGET_CKPT=/app/models/deploy/yolox_tiny_osc3.pth \
    CONTEXT_MODEL=on \
    TORCH_THREADS=1
EXPOSE 8001
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8001}"]
