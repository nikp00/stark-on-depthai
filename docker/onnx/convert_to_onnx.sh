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
python3 /stark-on-depthai/models/to_float_model/to_float_model.py
mv /stark-on-depthai/to_float_model*.onnx /stark-on-depthai/models/to_float_model/
python3 -m onnxsim /stark-on-depthai/models/to_float_model/to_float_model_128.onnx /stark-on-depthai/models/to_float_model/to_float_model_128_sim.onnx
python3 -m onnxsim /stark-on-depthai/models/to_float_model/to_float_model_320.onnx /stark-on-depthai/models/to_float_model/to_float_model_320_sim.onnx
