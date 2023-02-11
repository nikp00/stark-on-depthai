import depthai as dai
import numpy as np
import cv2
import struct
from pathlib import Path
from enum import Enum
import time

from typing import Optional, Callable, Tuple, List


## Logging setup
import logging

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)


class TrackerType(Enum):
    CAM = 0
    SYNTHETIC = 1


class Tracker:
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        script_node_path: str,
        type: TrackerType,
    ) -> None:
        self.type = type

        self.device = dai.Device()
        self.pipeline = self.build_pipeline(
            backbone_nn_path, complete_nn_path, script_node_path, type
        )
        self.device.startPipeline(self.pipeline)

        ## Input queues
        self.q_in_new_bbox = self.device.getInputQueue("in_new_bbox")
        self.q_in_debug_img_backbone = self.device.getInputQueue(
            "in_debug_img_backbone"
        )
        self.q_in_debug_img_complete = self.device.getInputQueue(
            "in_debug_img_complete"
        )
        self.q_in_debug_mask_z = self.device.getInputQueue("in_debug_mask_z")
        self.q_in_debug_feat_z = self.device.getInputQueue("in_debug_feat_z")
        self.q_in_debug_pos_z = self.device.getInputQueue("in_debug_pos_z")

        ## Output queues
        self.q_out_bbox = self.device.getOutputQueue("out_bbox", 1, False)
        self.q_out_img = self.device.getOutputQueue("out_img", 1, False)
        self.q_out_debug_crop = self.device.getOutputQueue("out_debug_crop", 1, False)
        self.q_out_debug_backbone_nn = self.device.getOutputQueue(
            "out_debug_backbone_nn", 1, False
        )

        # TODO
        self.mask_z = None
        self.feat_z = None
        self.pos_z = None

        ## Debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # self.device.setLogLevel(dai.LogLevel.DEBUG)
        # self.device.setLogOutputLevel(dai.LogLevel.DEBUG)

        self.logger.info("Tracker initialized")

    @staticmethod
    def read_script(script_base_path: str, script_name: str) -> str:
        with open(
            Path(script_base_path).resolve().joinpath("common.py"), "r"
        ) as script_file:
            common = script_file.read()

        with open(
            Path(script_base_path).resolve().joinpath(script_name), "r"
        ) as script_file:
            return script_file.read().replace("## Common", common)

    @staticmethod
    def buffer_to_list(buffer, dtype=int):
        if dtype == int:
            return struct.unpack("<" + "H" * (len(buffer) // 2), buffer)
        elif dtype == float:
            return struct.unpack("<" + "f" * (len(buffer) // 4), buffer)

    def build_pipeline(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        script_node_path: str,
        type: TrackerType,
    ) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        #########################################
        ##  Create nodes                      ##
        #########################################

        ## Initial resizer
        initial_resizer = pipeline.createImageManip()
        initial_resizer.inputConfig.setWaitForMessage(False)
        initial_resizer.initialConfig.setResize(640, 480)
        initial_resizer.initialConfig.setKeepAspectRatio(False)
        initial_resizer.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        ## Image source
        if type == TrackerType.CAM:
            img_source = pipeline.createColorCamera()
            img_source.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            img_source.setPreviewSize(640, 480)
            img_source.setInterleaved(False)
            img_source.setFps(30)

            # TODO: Check if this is needed
            # img_source.preview.link(initial_resizer.inputImage)

        elif type == TrackerType.SYNTHETIC:
            img_source = pipeline.createXLinkIn()
            img_source.setStreamName("in_img")
            img_source.setMaxDataSize(3 * 1920 * 1080)

            # TODO: Check if this is needed
            # img_source.out.link(initial_resizer.inputImage)

        ## XLinkIn for new bounding box
        new_bbox = pipeline.createXLinkIn()
        new_bbox.setStreamName("in_new_bbox")
        new_bbox.setMaxDataSize(16)

        # TODO
        ## XLinkIn for image
        xin_debug_img_complete = pipeline.createXLinkIn()
        xin_debug_img_complete.setStreamName("in_debug_img_complete")
        xin_debug_img_complete.setMaxDataSize(3 * 640 * 480)

        xin_debug_img_backbone = pipeline.createXLinkIn()
        xin_debug_img_backbone.setStreamName("in_debug_img_backbone")
        xin_debug_img_backbone.setMaxDataSize(3 * 640 * 480)

        xin_feat_z = pipeline.createXLinkIn()
        xin_feat_z.setStreamName("in_debug_feat_z")
        xin_feat_z.setMaxDataSize(3 * 640 * 640)

        xin_mask_z = pipeline.createXLinkIn()
        xin_mask_z.setStreamName("in_debug_mask_z")
        xin_mask_z.setMaxDataSize(640 * 640 * 3)

        xin_pos_z = pipeline.createXLinkIn()
        xin_pos_z.setStreamName("in_debug_pos_z")
        xin_pos_z.setMaxDataSize(640 * 640 * 3)

        ## Preprocessor script
        preprocessor_script = pipeline.create(dai.node.Script)
        preprocessor_script.setScript(
            Tracker.read_script(script_node_path, "preprocessor.py")
        )

        ## ImageManip for cropping
        crop = pipeline.createImageManip()
        crop.inputConfig.setWaitForMessage(True)
        crop.setMaxOutputFrameSize(3 * 640 * 480)
        crop.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        ## Router script
        router_script = pipeline.create(dai.node.Script)
        router_script.setScript(Tracker.read_script(script_node_path, "router.py"))

        ## Backbone NN
        backbone_nn = pipeline.createNeuralNetwork()
        backbone_nn.setBlobPath(Path(backbone_nn_path).resolve())

        ## Complete NN
        complete_nn = pipeline.createNeuralNetwork()
        complete_nn.setBlobPath(Path(complete_nn_path).resolve())

        ## Map box script
        map_bbox_script = pipeline.create(dai.node.Script)
        map_bbox_script.setScript(Tracker.read_script(script_node_path, "map_bbox.py"))

        ## XLinkOut for bounding box
        xout_bbox = pipeline.createXLinkOut()
        xout_bbox.setStreamName("out_bbox")

        ## XLinkOut for image
        xout_img = pipeline.createXLinkOut()
        xout_img.setStreamName("out_img")

        ## XLinkOut for debug crop
        xout_debug_crop = pipeline.createXLinkOut()
        xout_debug_crop.setStreamName("out_debug_crop")

        ## TODO
        ## XLinkOut for debug backbone NN
        xout_debug_backbone_nn = pipeline.createXLinkOut()
        xout_debug_backbone_nn.setStreamName("out_debug_backbone_nn")

        #########################################
        ##  Linking the nodes together         ##
        #########################################

        ## Preprocessor
        # TODO: Check if this is needed
        # initial_resizer.out.link(preprocessor_script.inputs["in_img"])
        img_source.out.link(preprocessor_script.inputs["in_img"])
        new_bbox.out.link(preprocessor_script.inputs["in_new_bbox"])
        map_bbox_script.outputs["out_bbox"].link(preprocessor_script.inputs["in_bbox"])

        ## Crop
        preprocessor_script.outputs["out_img"].link(crop.inputImage)
        preprocessor_script.outputs["out_cfg_crop"].link(crop.inputConfig)

        ## Router
        new_bbox.out.link(router_script.inputs["in_new_bbox"])
        crop.out.link(router_script.inputs["in_img"])
        preprocessor_script.outputs["out_mask"].link(router_script.inputs["in_mask"])
        backbone_nn.out.link(router_script.inputs["in_nn"])
        preprocessor_script.outputs["out_bbox"].link(router_script.inputs["in_bbox"])

        ## Backbone NN
        # TODO: Check if this is needed
        # router_script.outputs["out_img_z"].link(backbone_nn.inputs["img_z"])
        xin_debug_img_backbone.out.link(backbone_nn.inputs["img_z"])
        router_script.outputs["out_mask_z"].link(backbone_nn.inputs["mask_z"])

        ## Complete NN
        # TODO: Check if this is needed
        # router_script.outputs["out_img_x"].link(complete_nn.inputs["img_x"])
        xin_debug_img_complete.out.link(complete_nn.inputs["img_x"])
        router_script.outputs["out_mask_x"].link(complete_nn.inputs["mask_x"])
        # router_script.outputs["out_feat_z"].link(complete_nn.inputs["feat_vec_z"])
        # router_script.outputs["out_mask_z"].link(complete_nn.inputs["mask_vec_z"])
        # router_script.outputs["out_pos_z"].link(complete_nn.inputs["pos_vec_z"])
        xin_feat_z.out.link(complete_nn.inputs["feat_vec_z"])
        xin_mask_z.out.link(complete_nn.inputs["mask_vec_z"])
        xin_pos_z.out.link(complete_nn.inputs["pos_vec_z"])

        ## Map box
        complete_nn.out.link(map_bbox_script.inputs["in_nn"])
        preprocessor_script.outputs["out_resize_factor"].link(
            map_bbox_script.inputs["in_resize_factor"]
        )
        router_script.outputs["out_state"].link(map_bbox_script.inputs["in_state"])

        ## Outputs
        map_bbox_script.outputs["out_bbox"].link(xout_bbox.input)
        crop.out.link(xout_img.input)
        crop.out.link(xout_debug_crop.input)
        backbone_nn.out.link(xout_debug_backbone_nn.input)

        return pipeline


class SyntheticTracker(Tracker):
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        script_node_path: str,
    ) -> None:
        super().__init__(
            backbone_nn_path, complete_nn_path, script_node_path, TrackerType.SYNTHETIC
        )

        self.q_in_img = self.device.getInputQueue("in_img")

    def initialize(
        self, input_frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float], np.ndarray]:
        bbox_buffer = dai.Buffer()
        bbox_buffer.setData(list(struct.pack(f"<4f", *list(map(float, bbox)))))

        img_frame = dai.ImgFrame()
        img_frame.setWidth(input_frame.shape[1])
        img_frame.setHeight(input_frame.shape[0])
        img_frame.setData(input_frame.transpose(2, 0, 1).flatten())
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(time.monotonic())

        self.q_in_new_bbox.send(bbox_buffer)
        self.q_in_img.send(img_frame)

        img = self.q_out_img.get().getCvFrame()
        debug_crop = self.q_out_debug_crop.get().getCvFrame()

        img_buff = dai.Buffer()
        img_buff.setData(debug_crop.flatten().astype(np.float16).view(np.uint8))
        self.q_in_debug_img_backbone.send(img_buff)

        nn = self.q_out_debug_backbone_nn.get()
        self.feat_z = (
            np.array(nn.getLayerFp16("feat")).reshape((64, 1, 128)).astype(np.float32)
        )
        self.mask_z = np.array(nn.getLayerInt32("mask")).reshape((1, 64)).astype(bool)
        self.pos_z = np.array(nn.getLayerFp16("pos")).reshape((64, 1, 128))
        state = np.array(bbox)

        print(nn.getLayerFp16("feat")[:5])

        return img, bbox, debug_crop

    def track(
        self, input_frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float], np.ndarray]:
        img_frame = dai.ImgFrame()
        img_frame.setWidth(input_frame.shape[1])
        img_frame.setHeight(input_frame.shape[0])
        img_frame.setData(input_frame.transpose(2, 0, 1).flatten())
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(time.monotonic())

        self.q_in_img.send(img_frame)

        debug_crop = self.q_out_debug_crop.get().getCvFrame()
        img_buff = dai.Buffer()
        img_buff.setData(debug_crop.flatten().astype(np.float16).view(np.uint8))

        feat_z_buff = dai.Buffer()
        feat_z_buff.setData(self.feat_z.flatten().astype(np.float16).view(np.uint8))

        mask_z_buff = dai.Buffer()
        mask_z_buff.setData(self.mask_z.flatten().view(np.uint8))

        pos_z_buff = dai.Buffer()
        pos_z_buff.setData(self.pos_z.flatten().astype(np.float16).view(np.uint8))

        self.q_in_debug_img_complete.send(img_buff)
        self.q_in_debug_feat_z.send(feat_z_buff)
        self.q_in_debug_pos_z.send(pos_z_buff)
        self.q_in_debug_mask_z.send(mask_z_buff)

        bbox = self.buffer_to_list(self.q_out_bbox.get().getData(), float)
        img = self.q_out_img.get().getCvFrame()

        self.logger.info(f"bbox: {bbox}")

        return img, bbox, debug_crop
