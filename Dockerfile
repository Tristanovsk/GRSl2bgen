ARG IMAGE_SOURCE
FROM ${IMAGE_SOURCE}ubuntu:22.04 AS systemdependencies
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
COPY cert[s]/* /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Install libraries
RUN if [ -f "/kaniko/run/secrets/http_proxy" ]; then export http_proxy=$(cat /kaniko/run/secrets/http_proxy); export https_proxy=$(cat /kaniko/run/secrets/https_proxy); fi \
    && apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
        python-is-python3 \
        python3.10 \
        python3-pip \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# OBS2CO_L2BGEN INSTALL
SHELL ["/bin/sh", "-c"]
WORKDIR /home/
COPY obs2co_l2bgen /home/obs2co_l2bgen
COPY exe /home/obs2co_l2bgen/exe
COPY setup.py /home

# Add additionnal dependencies + L2BGEN
RUN if [ -f "/kaniko/run/secrets/http_proxy" ]; then export http_proxy=$(cat /kaniko/run/secrets/http_proxy); export https_proxy=$(cat /kaniko/run/secrets/https_proxy); fi \
    && pip3 install \
        --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
        --no-cache-dir \
        # requirements
        dask \
        docopt \
        matplotlib \
        netcdf4 \
        rioxarray \
        "xarray<=2023.4.2" \
    && pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org .

#ENTRYPOINT ["obs2co_l2bgen"]
