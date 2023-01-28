#!/bin/bash

docker build -t torch_17 $BASE_PATH/docker/onnx

docker run -it --rm \
--gpus all \
-v $BASE_PATH:/stark-on-depthai \
--workdir /stark-on-depthai \
--user "$(id -u):$(id -g)" \
torch_17
