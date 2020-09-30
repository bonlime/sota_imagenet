FROM nvcr.io/nvidia/pytorch:20.08-py3

# base setup
RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    wget software-properties-common pkg-config build-essential \
    libglu1-mesa -y && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

RUN pip --no-cache-dir install --upgrade \
    ConfigArgParse \
    ranger-fm \
    loguru

# install timm models library & inplace abn. to avoid building them each time later
RUN pip --no-cache-dir install --upgrade \
    git+https://github.com/rwightman/pytorch-image-models@master \
    inplace-abn==1.0.12 

# adding this VAR to force update of pytorch-tools each time
ARG RANDOM_VAR=none
RUN RANDOM_VAR=${RANDOM_VAR}
RUN pip --no-cache-dir install --upgrade \
    git+https://github.com/bonlime/pytorch-tools.git@add_darknet

COPY . /workdir
WORKDIR /workdir
