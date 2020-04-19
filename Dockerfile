# tensorflow-js@1.7.2 only supports CUDA=10.0
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

LABEL version="0.1"
LABEL description="PoseTrack working environment"

ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/cuda-10.0/bin:/opt/conda/bin/:${NVM_DIR}/bin/${PATH}
ENV NVM_DIR /opt/.nvm
ENV NODE_VERSION 12.16.2
ENV TFJS_VERSION 1.7.2
ENV POSENET_VERSION 2.2.1

# replace shell for node installation
RUN mv /bin/sh /bin/orgsh && ln -s /bin/bash /bin/sh

# apt install packages
RUN apt-get update && apt-get install -y \
    vim \
    wget

# install npm
RUN mkdir -p ${NVM_DIR}
RUN wget -qO- \ 
    https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash

# install node and node packages
RUN source ${NVM_DIR}/nvm.sh && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION}
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
RUN npm i \
    @tensorflow/tfjs-node-gpu@${TFJS_VERSION} \
    @tensorflow-models/posenet@${POSENET_VERSION} \
    canvas \
    express \
    body-parser

# create work directory
RUN mkdir -p /opt/workdir

# install torch-vision=0.4.2 (corresponding with pytorch=1.2)
WORKDIR /opt/workdir
RUN git clone https://github.com/pytorch/vision.git
WORKDIR /opt/workdir/vision
RUN git checkout v0.4.2
RUN python setup.py install

# install pretrained-models
WORKDIR /opt/workdir
RUN git clone https://github.com/Cadene/pretrained-models.pytorch.git
WORKDIR /opt/workdir/pretrained-models.pytorch/
RUN python setup.py install

# clone poseTrack
WORKDIR /opt/workdir
RUN git clone https://github.com/windsor718/poseTrack.git
RUN ~ ${HOME}/.bashrc
