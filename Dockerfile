# Project: Neural Style Transfer
# Author: Artan Zandian
# Interactive Usage:  docker run --rm -p 8888:8888 -v /"$(pwd)"://home//jovyan//work <image_name>
# Make Usage: docker run --rm -v /"$(pwd)"://home//jovyan//work artanzandian/
# Docker-compose Usage: docker-compose run --rm report-env make -C //home//jovyan//work all

FROM nvcr.io/nvidia/l4t-base:r32.6.1

# install python packages
WORKDIR /
RUN apt update && apt install -y --fix-missing \
    make \
    g++ \
    python3-pip \
    libhdf5-serial-dev \
    hdf5-tools \
RUN apt update && apt install -y \
    python3-h5py \
    pillow

# Install Tensor packages   
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install -U numpy
