# Project: Neural Style Transfer
# Author: Artan Zandian
# Interactive Usage:  docker run --rm -p 8888:8888 -v /"$(pwd)"://home//jovyan//work <image_name>
# Make Usage: docker run --rm -v /"$(pwd)"://home//jovyan//work artanzandian/canadian_heritage_funding make -C //home//jovyan//work all
# Docker-compose Usage: docker-compose run --rm report-env make -C //home//jovyan//work all

FROM jupyter/minimal-notebook


# Install Python 3 packages
RUN conda install -c conda-forge --quiet -y \
    python==3.9.* \
    tensorflow==2.6.0 \
    keras==2.6.* \
    pillow \
    numpy==1.22.* \
    # matplotlib
    # ipykernel
    # jupyter notebook

