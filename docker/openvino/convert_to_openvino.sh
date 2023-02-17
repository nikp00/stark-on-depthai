#!/bin/bash

mo \
--input_model /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.onnx \
--output_dir /stark-on-depthai/models/backbone_bottleneck_pe \
--framework onnx \
--input img,mask \
--input_shape [1,3,128,128],[128,128] \
--output feat_z,mask_z,pos_z \
--data_type=FP16 \
--log_level=DEBUG

mo \
--input_model /stark-on-depthai/models/complete/complete_sim.onnx \
--output_dir /stark-on-depthai/models/complete \
--framework onnx \
--input img_x,mask_x,feat_z,mask_z,pos_z \
--input_shape [1,3,320,320],[320,320],[64,1,128],[1,64],[64,1,128] \
--output outputs_coord \
--data_type=FP16 \
--log_level=DEBUG

mo \
--input_model /stark-on-depthai/models/complete/complete.onnx \
--output_dir /stark-on-depthai/models/complete \
--framework onnx \
--input img_x,mask_x,feat_z,mask_z,pos_z \
--input_shape [1,3,320,320],[320,320],[64,1,128],[1,64],[64,1,128] \
--output outputs_coord \
--data_type=FP16 \
--log_level=DEBUG

mo \
--input_model /stark-on-depthai/models/pre_model/pre_model_128_nn.onnx \
--output_dir /stark-on-depthai/models/pre_model \
--framework onnx \
--input in_img \
--input_shape [1,3,128,128] \
--output img \
--data_type=FP16 \
--log_level=DEBUG

mo \
--input_model /stark-on-depthai/models/pre_model/pre_model_320_nn.onnx \
--output_dir /stark-on-depthai/models/pre_model \
--framework onnx \
--input in_img \
--input_shape [1,3,320,320] \
--output img_x \
--data_type=FP16 \
--log_level=DEBUG