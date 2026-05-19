# syntax=docker/dockerfile:1.7

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
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
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
    libeigen3-dev \
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
    transformers==4.57.2 \
    xformers==0.0.29.post3

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

COPY o-voxel /workspace/o-voxel

RUN mkdir -p /workspace/o-voxel/third_party/eigen && \
    ln -sfn /usr/include/eigen3/Eigen /workspace/o-voxel/third_party/eigen/Eigen

RUN python -m pip install ./o-voxel --no-build-isolation

COPY assets /workspace/assets
COPY configs /workspace/configs
COPY trellis2 /workspace/trellis2
COPY app.py /workspace/app.py
COPY app_texturing.py /workspace/app_texturing.py
COPY example.py /workspace/example.py
COPY example_texturing.py /workspace/example_texturing.py
COPY runpod_handler.py /workspace/runpod_handler.py
COPY runpod_inference.py /workspace/runpod_inference.py
COPY runpod_request.example.json /workspace/runpod_request.example.json
COPY README.md /workspace/README.md
COPY LICENSE /workspace/LICENSE
COPY SECURITY.md /workspace/SECURITY.md

CMD ["python", "-u", "runpod_handler.py"]
