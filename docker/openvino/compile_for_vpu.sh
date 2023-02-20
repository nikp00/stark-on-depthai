#!/bin/bash

## Backbone
/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob \
-ip FP16 \
-VPU_NUMBER_OF_SHAVES 1 \
-VPU_NUMBER_OF_CMX_SLICES 1

## Complete simplified
/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/complete/complete_sim.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/complete/complete_myriad_sim.blob \
-ip FP16 \
-VPU_NUMBER_OF_SHAVES 8 \
-VPU_NUMBER_OF_CMX_SLICES 8

# ## Complete
# /opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
# -m /stark-on-depthai/models/complete/complete.xml \
# -d MYRIAD \
# -c /stark-on-depthai/docker/openvino/myriad.conf \
# -o /stark-on-depthai/models/complete/complete_myriad.blob \
# -ip FP16 \
# -VPU_NUMBER_OF_SHAVES 4 \
# -VPU_NUMBER_OF_CMX_SLICES 4

## To float model 128
/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/to_float_model/to_float_model_128.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/to_float_model/to_float_model_128.blob \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 1 \
-VPU_NUMBER_OF_CMX_SLICES 1

## To float model 320
/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/to_float_model/to_float_model_320.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/to_float_model/to_float_model_320.blob \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 1 \
-VPU_NUMBER_OF_CMX_SLICES 1