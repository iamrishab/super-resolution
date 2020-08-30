FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         git \
         cmake \
         gcc \
         g++ \
         libsm6 \
         libglib2.0-0 \
         libxrender-dev \
         libxext6 \
         libgl1-mesa-glx \
         sudo \
         gnupg2 \
         wget && \
 rm -rf /var/lib/apt/lists/*

COPY . /root/super-res/
WORKDIR /root/super-res/

RUN pip install -q --upgrade pip
RUN pip install -qr requirements.txt --ignore-installed

EXPOSE 8003
ENTRYPOINT ["python3", "api.py"]
