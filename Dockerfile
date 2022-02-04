# Project: Neural Style Transfer
# Author: Artan Zandian
# Interactive Usage:  docker run --rm -p 8888:8888 -v /"$(pwd)"://home//jovyan//work <image_name>
# Docker-compose Usage: docker-compose run --rm report-env make -C //home//jovyan//work all

# podman pull nvcr.io/nvidia/l4t-base:r32.2
# podman build --tag docker.io/artanzandian/keras:0.1.0 . -f ./Dockerfile
# podman push docker.io/artanzandian/keras:0.1.0

FROM nvcr.io/nvidia/l4t-base:r32.2

# install python packages
WORKDIR /
RUN apt update && apt install -y --fix-missing make g++
RUN apt update && apt install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt update && apt install -y python3-h5py

# install tensor related packages
RUN pip3 install --upgrade setuptools
RUN pip3 install cython pillow docopt
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install -U numpy


