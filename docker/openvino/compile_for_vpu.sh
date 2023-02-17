#!/bin/bash

/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob \
-ip FP16 \
-VPU_NUMBER_OF_SHAVES 6 \
-VPU_NUMBER_OF_CMX_SLICES 6

/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/complete/complete_sim.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/complete/complete_myriad_sim.blob \
-ip FP16 \
-VPU_NUMBER_OF_SHAVES 7 \
-VPU_NUMBER_OF_CMX_SLICES 7

/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/complete/complete.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/complete/complete_myriad.blob \
-ip FP16 \
-VPU_NUMBER_OF_SHAVES 6 \
-VPU_NUMBER_OF_CMX_SLICES 6

/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/pre_model/pre_model_128_nn.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/pre_model/pre_model_128_nn.blob \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 6 \
-VPU_NUMBER_OF_CMX_SLICES 6

/opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool \
-m /stark-on-depthai/models/pre_model/pre_model_320_nn.xml \
-d MYRIAD \
-c /stark-on-depthai/docker/openvino/myriad.conf \
-o /stark-on-depthai/models/pre_model/pre_model_320_nn.blob \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 6 \
-VPU_NUMBER_OF_CMX_SLICES 6