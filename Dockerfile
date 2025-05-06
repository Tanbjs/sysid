# Base image with CUDA 12.5 and cuDNN 9.3
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# set the locale
USER root
# reduce interactive prompt เวลา apt install
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies and clean cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      wget \
      python3 \
      python3-pip \
      python3-dev \
      python3-setuptools \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

# create file system for thesis_ws
WORKDIR /home/thesis_ws/sysid

# Copy only requirements.txt to cache layer of pip install
COPY . /home/thesis_ws/sysid/

# upgrade pip then install Python dependencies
RUN pip install --upgrade pip

# install Python dependencies
RUN pip install -r requirements.txt

# use bash as default shell
CMD ["bash"]
