ARG IMAGE_SOURCE
# Ubuntu 24.04: Python 3.12 (GRSl2bgen needs Python >= 3.11, as zarr 3)
FROM ${IMAGE_SOURCE}ubuntu:24.04 AS systemdependencies
#FROM ubuntu:kinetic AS systemdependencies
LABEL maintainer: "robin.buratti@magellium.fr"

ENV LANG C.UTF-8
ENV LC_ C.UTF-8

RUN ulimit -s unlimited

# Proxy from secret volumes
RUN if [ -f "/kaniko/run/secrets/http_proxy" ]; then export http_proxy=$(cat /kaniko/run/secrets/http_proxy); export https_proxy=$(cat /kaniko/run/secrets/https_proxy); fi && \
    apt-get update -y && \
    apt-get install -y ca-certificates

# Ajout des certificats
RUN update-ca-certificates

# Install libraries
RUN if [ -f "/kaniko/run/secrets/http_proxy" ]; then export http_proxy=$(cat /kaniko/run/secrets/http_proxy); export https_proxy=$(cat /kaniko/run/secrets/https_proxy); fi \
    && apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
        python3 \
        python3-venv \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# GRSL2BGEN INSTALL
SHELL ["/bin/sh", "-c"]
WORKDIR /home/
COPY GRSl2bgen /opt/GRSl2bgen/GRSl2bgen
COPY exe /opt/GRSl2bgen/exe
COPY pyproject.toml requirements.txt README.md LICENSE /opt/GRSl2bgen/

# the system Python of Ubuntu 24.04 refuses pip installs (PEP 668): virtual environment,
# first in the PATH so that python, pip and GRSl2bgen are the ones of the environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Add additionnal dependencies + L2BGEN
RUN python3 -m venv /opt/venv \
    && if [ -f "/kaniko/run/secrets/http_proxy" ]; then export http_proxy=$(cat /kaniko/run/secrets/http_proxy); export https_proxy=$(cat /kaniko/run/secrets/https_proxy); fi \
    && pip install \
        --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
        --no-cache-dir --upgrade pip setuptools \
    # dependencies come from requirements.txt, as in the CI tests
    && pip install \
        --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
        --no-cache-dir /opt/GRSl2bgen

#ENTRYPOINT ["GRSl2bgen"]
