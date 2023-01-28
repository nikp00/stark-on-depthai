#!/bin/bash

mo \
--input_model /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.onnx \
--output_dir /stark-on-depthai/models/backbone_bottleneck_pe \
--framework onnx \
--input img_z,mask_z \
--input_shape [1,3,128,128],[1,128,128] \
--output feat,mask,pos \
--data_type=FP16 \
--log_level=DEBUG

mo \
--input_model /stark-on-depthai/models/complete/complete_sim.onnx \
--output_dir /stark-on-depthai/models/complete \
--framework onnx \
--input img_x,feat_vec_z,mask_vec_z,pos_vec_z \
--input_shape [1,3,320,320],[64,1,128],[1,64],[64,1,128] \
--output outputs_coord \
--data_type=FP16 \
--log_level=DEBUG