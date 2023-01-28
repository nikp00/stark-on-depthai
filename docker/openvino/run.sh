#!/bin/bash

docker run -it --rm \
-v $BASE_PATH:/stark-on-depthai \
--device /dev/dri:/dev/dri \
--device-cgroup-rule='c 189:* rmw' \
-v /dev/bus/usb:/dev/bus/usb \
--workdir /stark-on-depthai \
openvino/ubuntu20_dev:latest