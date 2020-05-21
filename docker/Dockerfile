FROM nvcr.io/nvidia/pytorch:20.03-py3

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
    # git+https://github.com/bonlime/pytorch-tools.git@v0.1.4 \
    git+https://github.com/bonlime/pytorch-tools.git@dev \
    loguru

COPY . /workdir
WORKDIR /workdir
