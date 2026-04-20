FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=xformers \
    SPARSE_ATTN_BACKEND=xformers \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/runpod-volume/huggingface \
    TORCH_HOME=/runpod-volume/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && python -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --retries 10 --timeout 120 \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --retries 10 --timeout 120 --prefer-binary \
    numpy==1.26.4 \
    scipy==1.15.3 \
    pandas==2.2.3 \
    python-dateutil==2.9.0.post0 \
    six \
    pytz \
    tzdata

RUN python -m pip install --retries 10 --timeout 120 --prefer-binary \
    easydict \
    imageio \
    imageio-ffmpeg \
    ninja \
    opencv-python-headless \
    requests \
    runpod \
    tensorboard \
    tqdm \
    trimesh \
    zstandard

RUN python -m pip install --retries 10 --timeout 120 --prefer-binary \
    kornia \
    lpips \
    timm \
    transformers \
    xformers

RUN python -m pip install --retries 10 --timeout 120 \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN python -m pip install --retries 10 --timeout 120 --no-build-isolation flash-attn==2.7.3 || true

RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    python -m pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    python -m pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    python -m pip install /tmp/extensions/CuMesh --no-build-isolation

RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    python -m pip install /tmp/extensions/FlexGEMM --no-build-isolation

RUN python -m pip install ./o-voxel --no-build-isolation

CMD ["python", "-u", "runpod_handler.py"]
