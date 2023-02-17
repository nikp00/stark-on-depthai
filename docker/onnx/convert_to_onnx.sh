#!/bin/bash

python3 Stark/tracking/create_default_local_file.py --workspace_dir ./Stark/ --data_dir ./Stark/data --save_dir ./Stark

cd Stark

mkdir -p /stark-on-depthai/models

python3 /stark-on-depthai/Stark/tracking/ORT_lightning_X_trt_backbone_bottleneck_pe.py
mkdir -p /stark-on-depthai/models/backbone_bottleneck_pe
mv /stark-on-depthai/Stark/backbone*.onnx /stark-on-depthai/models/backbone_bottleneck_pe/
python3 -m onnxsim /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.onnx /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe_sim.onnx


python3 /stark-on-depthai/Stark/tracking/ORT_lightning_X_trt_complete.py
mkdir -p /stark-on-depthai/models/complete
mv /stark-on-depthai/Stark/complete*.onnx /stark-on-depthai/models/complete/
mv /stark-on-depthai/Stark/complete*.pth /stark-on-depthai/models/complete/
python3 -m onnxsim /stark-on-depthai/models/complete/complete.onnx /stark-on-depthai/models/complete/complete_sim.onnx


cd ..
python3 /stark-on-depthai/models/pre_model/pre_model.py
mv /stark-on-depthai/pre_model*.onnx /stark-on-depthai/models/pre_model/
python3 -m onnxsim /stark-on-depthai/models/pre_model/pre_model_128_nn.onnx /stark-on-depthai/models/pre_model/pre_model_128_nn_sim.onnx
python3 -m onnxsim /stark-on-depthai/models/pre_model/pre_model_320_nn.onnx /stark-on-depthai/models/pre_model/pre_model_320_nn_sim.onnx
