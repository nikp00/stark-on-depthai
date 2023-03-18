import depthai as dai
import numpy as np
import cv2
import struct
from pathlib import Path
from enum import Enum
import time
import math
from .utils import FpsCounter

from typing import Optional, Callable, Tuple, List, Union


## Logging setup
import logging

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)


class TrackerType(Enum):
    CAM = 0
    SYNTHETIC = 1


class EdgeModeTracker:
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
        script_node_path: str,
        type: TrackerType,
        debug: bool = False,
        script_debug_level: int = -1,
    ) -> None:
        self.debug = debug
        self.type = type
        self.script_debug_level = script_debug_level

        self.fps = FpsCounter()

        self.device = dai.Device()
        self.pipeline = self.build_pipeline(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
            script_node_path,
        )
        self.device.startPipeline(self.pipeline)

        ## Input queues
        self.q_in_new_bbox = self.device.getInputQueue("in_new_bbox")

        ## Output queues
        self.q_out_bbox = self.device.getOutputQueue("out_bbox", 1, False)
        self.q_out_resized_img = self.device.getOutputQueue("out_resized_img", 1, False)
        self.q_out_img = self.device.getOutputQueue("out_img", 1, False)

        #############
        ##  Debug  ##
        #############
        if self.debug:
            self.device.setLogLevel(dai.LogLevel.DEBUG)
            self.device.setLogOutputLevel(dai.LogLevel.DEBUG)
        else:
            self.device.setLogLevel(dai.LogLevel.WARN)
            self.device.setLogOutputLevel(dai.LogLevel.WARN)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Tracker initialized")

        if self.debug:
            self.logger.info("Debug mode enabled")
            self.q_out_debug_crop = self.device.getOutputQueue(
                "out_debug_crop", 1, False
            )
            self.q_out_debug_float = self.device.getOutputQueue(
                "out_debug_float", 1, False
            )

    def read_script(self, script_base_path: str, script_name: str) -> str:
        with open(
            Path(script_base_path).resolve().joinpath(script_name), "r"
        ) as script_file:
            return script_file.read().replace(
                "LOG_LEVEL = -1", f"LOG_LEVEL = {self.script_debug_level}"
            )

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
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
        script_node_path: str,
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
        if self.type == TrackerType.CAM:
            img_source = pipeline.createColorCamera()
            img_source.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            img_source.setPreviewSize(1920, 1080)
            img_source.setInterleaved(False)
            img_source.setFps(10)

            img_source.preview.link(initial_resizer.inputImage)
            img_source_out = img_source.preview

        elif self.type == TrackerType.SYNTHETIC:
            img_source = pipeline.createXLinkIn()
            img_source.setStreamName("in_img")
            img_source.setMaxDataSize(3 * 1920 * 1080)

            img_source.out.link(initial_resizer.inputImage)
            img_source_out = img_source.out

        ## XLinkIn for new bounding box
        new_bbox = pipeline.createXLinkIn()
        new_bbox.setStreamName("in_new_bbox")
        new_bbox.setMaxDataSize(4 * 4)

        ## XLinkOut for bounding box
        out_bbox = pipeline.createXLinkOut()
        out_bbox.setStreamName("out_bbox")

        ## XLinkOut for resized image
        out_resized_img = pipeline.createXLinkOut()
        out_resized_img.setStreamName("out_resized_img")

        ## XLinkOut for image
        out_img = pipeline.createXLinkOut()
        out_img.setStreamName("out_img")

        ## Manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(
            self.read_script(script_node_path, "manager_script.py")
        )
        if self.type == TrackerType.CAM:
            manager_script.inputs["in_img"].setQueueSize(1)
            manager_script.inputs["in_img"].setBlocking(False)
            manager_script.inputs["in_resized_img"].setQueueSize(1)
            manager_script.inputs["in_resized_img"].setBlocking(False)

        ## Pre-backbone crop
        pre_backbone_crop = pipeline.create(dai.node.ImageManip)
        pre_backbone_crop.inputConfig.setWaitForMessage(True)
        pre_backbone_crop.initialConfig.setResize(128, 128)
        pre_backbone_crop.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        ## Pre-complete crop
        pre_complete_crop = pipeline.create(dai.node.ImageManip)
        pre_complete_crop.inputConfig.setWaitForMessage(True)
        pre_complete_crop.initialConfig.setResize(320, 320)
        pre_complete_crop.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        ## Pre-model backbone
        to_float_backbone_nn = pipeline.create(dai.node.NeuralNetwork)
        to_float_backbone_nn.setBlobPath(pre_backbone_nn_path)

        ## Pre-model complete
        to_float_complete_nn = pipeline.create(dai.node.NeuralNetwork)
        to_float_complete_nn.setBlobPath(pre_complete_nn_path)

        ## Backbone NN
        backbone_nn = pipeline.create(dai.node.NeuralNetwork)
        backbone_nn.setBlobPath(backbone_nn_path)

        ## Complete NN
        complete_nn = pipeline.create(dai.node.NeuralNetwork)
        complete_nn.setBlobPath(complete_nn_path)
        # complete_nn.setNumInferenceThreads(1)  # By default 2 threads are used
        # complete_nn.setNumNCEPerInferenceThread(2)

        #####################
        ##  Linking nodes  ##
        #####################
        initial_resizer.out.link(manager_script.inputs["in_resized_img"])
        initial_resizer.out.link(out_resized_img.input)
        img_source_out.link(manager_script.inputs["in_img"])
        img_source_out.link(out_img.input)
        new_bbox.out.link(manager_script.inputs["in_new_bbox"])

        manager_script.outputs["out_img_backbone"].link(pre_backbone_crop.inputImage)
        manager_script.outputs["out_cfg_crop_backbone"].link(
            pre_backbone_crop.inputConfig
        )
        pre_backbone_crop.out.link(to_float_backbone_nn.inputs["in_img"])
        to_float_backbone_nn.out.link(backbone_nn.input)
        backbone_nn.out.link(manager_script.inputs["in_backbone_result"])

        manager_script.outputs["out_img_complete"].link(pre_complete_crop.inputImage)
        manager_script.outputs["out_cfg_crop_complete"].link(
            pre_complete_crop.inputConfig
        )
        pre_complete_crop.out.link(to_float_complete_nn.inputs["in_img"])
        to_float_complete_nn.out.link(complete_nn.inputs["img_x"])
        manager_script.outputs["out_backbone_result"].link(complete_nn.input)
        complete_nn.out.link(manager_script.inputs["in_complete_result"])

        manager_script.outputs["out_bbox"].link(out_bbox.input)

        #################
        ##  Debugging  ##
        #################
        if self.debug:
            xout_debug_crop = pipeline.create(dai.node.XLinkOut)
            xout_debug_crop.setStreamName("out_debug_crop")
            pre_backbone_crop.out.link(xout_debug_crop.input)
            pre_complete_crop.out.link(xout_debug_crop.input)

            xout_debug_float = pipeline.create(dai.node.XLinkOut)
            xout_debug_float.setStreamName("out_debug_float")
            to_float_backbone_nn.out.link(xout_debug_float.input)
            to_float_complete_nn.out.link(xout_debug_float.input)

            xin_debug_img_backbone = pipeline.create(dai.node.XLinkIn)
            xin_debug_img_backbone.setStreamName("in_debug_img_backbone")

            xin_debug_img_complete = pipeline.create(dai.node.XLinkIn)
            xin_debug_img_complete.setStreamName("in_debug_img_complete")

        return pipeline


class EdgeModeSyntheticTracker(EdgeModeTracker):
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
        script_node_path: str,
        debug: bool = False,
        script_debug_level: int = -1,
    ) -> None:
        super().__init__(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
            script_node_path,
            TrackerType.SYNTHETIC,
            debug,
            script_debug_level,
        )

        self.q_in_img = self.device.getInputQueue("in_img")

    def initialize(
        self, input_frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        self.fps.reset()
        self.fps.start()

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

        out_img = self.q_out_img.get().getCvFrame()
        resized_img = self.q_out_resized_img.get().getCvFrame()

        if self.debug:
            cropped = self.q_out_debug_crop.get().getCvFrame()
            debug_float = np.array(
                self.q_out_debug_float.get().getLayerFp16("img"), dtype=np.float16
            )
            cv2.imshow(
                "Converted to float",
                debug_float.reshape(3, 128, 128).transpose(1, 2, 0).astype(np.uint8),
            )

            cv2.imshow("Cropped ROI", cropped)

        self.fps.tick()

        return out_img, bbox

    def track(
        self, input_frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        img_frame = dai.ImgFrame()
        img_frame.setWidth(input_frame.shape[1])
        img_frame.setHeight(input_frame.shape[0])
        img_frame.setData(input_frame.transpose(2, 0, 1).flatten())
        img_frame.setType(dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(time.monotonic())

        self.q_in_img.send(img_frame)

        if self.debug:
            debug_float = np.array(
                self.q_out_debug_float.get().getLayerFp16("img_x"), dtype=np.float16
            )

            cropped = self.q_out_debug_crop.get().getCvFrame()

            cv2.imshow(
                "Converted to float",
                debug_float.reshape(3, 320, 320).transpose(1, 2, 0).astype(np.uint8),
            )

            cv2.imshow(
                "Cropped ROI",
                cropped,
            )

        bbox = self.buffer_to_list(self.q_out_bbox.get().getData(), float)
        resized_img = self.q_out_resized_img.get().getCvFrame()
        out_img = self.q_out_img.get().getCvFrame()

        self.fps.tick()
        fps, avg_fps = self.fps.get()
        self.logger.info(f"Avg FPS: {avg_fps}  |  FPS: {fps}  |  bbox: {bbox}")

        return out_img, bbox


class EdgeModeCamTracker(EdgeModeTracker):
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
        script_node_path: str,
        debug: bool = False,
        script_debug_level: int = -1,
    ) -> None:
        super().__init__(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
            script_node_path,
            TrackerType.CAM,
            debug,
            script_debug_level,
        )

        self.ix = 0
        self.iy = 0
        self.iw = 0
        self.ih = 0
        self.mouse_pressed = False
        self.initialize_tracker = False
        self.initialized = False

    def new_bbox_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.ix, self.iy = x, y
            self.initialized = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                self.iw, self.ih = x - self.ix, y - self.iy
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            self.initialize_tracker = True
            self.logger.info(f"New bbox: {self.ix, self.iy, self.iw, self.ih}")
            if self.iw < 0:
                self.ix += self.iw
                self.iw = abs(self.iw)
            if self.ih < 0:
                self.iy += self.ih
                self.ih = abs(self.ih)
            self.logger.info(f"New bbox: {self.ix, self.iy, self.iw, self.ih}")

    def run(self):
        if self.initialize_tracker and not self.initialized:
            self.initialize_tracker = False
            self.initialized = True
            out_img, bbox = self.initialize((self.ix, self.iy, self.iw, self.ih))
        elif self.initialized:
            out_img, bbox = self.track()
        else:
            out_img = self.q_out_img.get().getCvFrame()
            bbox = [0, 0, out_img.shape[1], out_img.shape[0]]

        return out_img, bbox

    def initialize(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        self.fps.reset()
        self.fps.start()

        bbox_buffer = dai.Buffer()
        bbox_buffer.setData(list(struct.pack(f"<4f", *list(map(float, bbox)))))

        self.q_in_new_bbox.send(bbox_buffer)

        if self.debug:
            cropped = self.q_out_debug_crop.get().getCvFrame()
            debug_float = np.array(
                self.q_out_debug_float.get().getLayerFp16("img"), dtype=np.float16
            )

            if debug_float.size > 0:
                cv2.imshow(
                    "Converted to float",
                    debug_float.reshape(3, 128, 128)
                    .transpose(1, 2, 0)
                    .astype(np.uint8),
                )

                cv2.imshow(
                    "Cropped ROI",
                    cropped,
                )

        out_img = self.q_out_img.get().getCvFrame()

        self.fps.tick()

        return out_img, bbox

    def track(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        timer_start = time.time() * 1000
        if self.debug:
            debug_float = np.array(
                self.q_out_debug_float.get().getLayerFp16("img_x"), dtype=np.float16
            )

            cropped = self.q_out_debug_crop.get().getCvFrame()
            if debug_float.size > 0:
                cv2.imshow(
                    "Converted to float",
                    debug_float.reshape(3, 320, 320)
                    .transpose(1, 2, 0)
                    .astype(np.uint8),
                )

                cv2.imshow(
                    "Cropped ROI",
                    cropped,
                )

        bbox = self.buffer_to_list(self.q_out_bbox.get().getData(), float)
        resized_img = self.q_out_resized_img.get().getCvFrame()
        out_img = self.q_out_img.get().getCvFrame()

        self.fps.tick()
        fps, avg_fps = self.fps.get()
        self.logger.info(f"Avg FPS: {avg_fps}  |  FPS: {fps}  |  bbox: {bbox}")
        self.logger.info(f"{time.time() * 1000 - timer_start}")

        return out_img, bbox


class HostModeTracker:
    TEMPLATE_SIZE = 128
    TEMPLATE_FACTOR = 2.0
    SEARCH_SIZE = 320
    SEARCH_FACTOR = 5.0

    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
        type: TrackerType,
    ) -> None:
        self.type = type

        self.fps = FpsCounter()

        self.device = dai.Device()
        self.pipeline = self.build_pipeline(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
        )
        self.device.startPipeline(self.pipeline)

        ## Input queues
        self.q_in_backbone_img = self.device.getInputQueue("in_backbone_img")
        # self.q_in_backbone_mask = self.device.getInputQueue("in_backbone_mask")

        self.q_in_complete_img_x = self.device.getInputQueue("in_complete_img_x")
        # self.q_in_complete_mask_x = self.device.getInputQueue("in_complete_mask_x")
        self.q_in_complete_feat_z = self.device.getInputQueue("in_complete_feat_z")
        self.q_in_complete_mask_z = self.device.getInputQueue("in_complete_mask_z")
        self.q_in_complete_pos_z = self.device.getInputQueue("in_complete_pos_z")

        ## Output queues
        self.q_out_img = self.device.getOutputQueue("out_img", 1, False)
        self.q_out_backbone = self.device.getOutputQueue("out_backbone", 1, False)
        self.q_out_complete = self.device.getOutputQueue("out_complete", 1, False)

        self.feat_z = None
        self.mask_z = None
        self.pos_z = None
        self.state = None

        #############
        ##  Debug  ##
        #############
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Tracker initialized")

    def build_pipeline(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
    ) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        #########################################
        ##  Create nodes                      ##
        #########################################

        ## XLinkIn
        xlink_in_backbone_img = pipeline.createXLinkIn()
        xlink_in_backbone_img.setStreamName("in_backbone_img")
        xlink_in_backbone_img.setMaxDataSize(3 * 320 * 320 * 8)

        xlink_in_backbone_mask = pipeline.createXLinkIn()
        xlink_in_backbone_mask.setStreamName("in_backbone_mask")
        xlink_in_backbone_mask.setMaxDataSize(320 * 320)

        xlink_in_complete_img_x = pipeline.createXLinkIn()
        xlink_in_complete_img_x.setStreamName("in_complete_img_x")
        xlink_in_complete_img_x.setMaxDataSize(3 * 320 * 320 * 8)

        xlink_in_complete_feat_z = pipeline.createXLinkIn()
        xlink_in_complete_feat_z.setStreamName("in_complete_feat_z")
        xlink_in_complete_feat_z.setMaxDataSize(64 * 128 * 8)

        xlink_in_complete_mask_z = pipeline.createXLinkIn()
        xlink_in_complete_mask_z.setStreamName("in_complete_mask_z")
        xlink_in_complete_mask_z.setMaxDataSize(64 * 8)

        xlink_in_complete_pos_z = pipeline.createXLinkIn()
        xlink_in_complete_pos_z.setStreamName("in_complete_pos_z")
        xlink_in_complete_pos_z.setMaxDataSize(64 * 128 * 8)

        ## XLinkOut
        xlink_out_img = pipeline.createXLinkOut()
        xlink_out_img.setStreamName("out_img")

        xlink_out_backbone = pipeline.createXLinkOut()
        xlink_out_backbone.setStreamName("out_backbone")

        xlink_out_complete = pipeline.createXLinkOut()
        xlink_out_complete.setStreamName("out_complete")

        ## Image source
        if self.type == TrackerType.CAM:
            img_source = pipeline.createColorCamera()
            img_source.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            img_source.setPreviewSize(640, 480)
            img_source.setInterleaved(False)
            img_source.setFps(30)

            img_source.preview.link(xlink_out_img.input)

        ## Pre-model backbone
        pre_backbone_nn = pipeline.create(dai.node.NeuralNetwork)
        pre_backbone_nn.setBlobPath(pre_backbone_nn_path)

        ## Pre-model complete
        pre_complete_nn = pipeline.create(dai.node.NeuralNetwork)
        pre_complete_nn.setBlobPath(pre_complete_nn_path)

        ## Backbone NN
        backbone_nn = pipeline.create(dai.node.NeuralNetwork)
        backbone_nn.setBlobPath(backbone_nn_path)

        ## Complete NN
        complete_nn = pipeline.create(dai.node.NeuralNetwork)
        complete_nn.setBlobPath(complete_nn_path)

        #####################
        ##  Linking nodes  ##
        #####################
        xlink_in_backbone_img.out.link(pre_backbone_nn.inputs["in_img"])
        pre_backbone_nn.out.link(backbone_nn.input)

        backbone_nn.out.link(xlink_out_backbone.input)

        xlink_in_complete_img_x.out.link(pre_complete_nn.inputs["in_img"])
        pre_complete_nn.out.link(complete_nn.input)
        xlink_in_complete_feat_z.out.link(complete_nn.inputs["feat_z"])
        xlink_in_complete_mask_z.out.link(complete_nn.inputs["mask_z"])
        xlink_in_complete_pos_z.out.link(complete_nn.inputs["pos_z"])

        complete_nn.out.link(xlink_out_complete.input)

        return pipeline

    def sample_target(
        self,
        im: np.ndarray,
        bbox: list,
        search_area_factor: float,
        output_sz: Union[int, float],
    ):
        x, y, w, h = bbox

        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception("Too small bounding box.")

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(
            im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT
        )
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        return im_crop_padded, resize_factor

    def clip_bbox(self, box: np.ndarray, H: int, W: int, margin: int = 0) -> np.ndarray:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return np.array([x1, y1, w, h])

    def map_box_back(
        self,
        pred_box: np.ndarray,
        state: np.ndarray,
        resize_factor: float,
        search_size: float,
    ) -> list:
        cx_prev, cy_prev = (
            state[0] + 0.5 * state[2],
            state[1] + 0.5 * state[3],
        )
        cx, cy, w, h = pred_box
        half_side = 0.5 * search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


class HostModeSyntheticTracker(HostModeTracker):
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
    ):
        super().__init__(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
            TrackerType.SYNTHETIC,
        )

    def initialize(
        self, input_frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float], np.ndarray]:
        self.logger.info("Initializing tracker...")
        self.fps.reset()
        self.fps.start()

        img, _ = self.sample_target(
            input_frame, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE
        )

        img_buff = dai.Buffer()
        img_buff.setData(
            img.transpose(2, 0, 1).flatten().astype(np.uint8).view(np.uint8)
        )

        self.q_in_backbone_img.send(img_buff)

        backbone_out = self.q_out_backbone.get()

        self.feat_z = (
            np.array(backbone_out.getLayerFp16("feat_z"))
            .reshape((64, 1, 128))
            .astype(np.float32)
        )

        self.mask_z = (
            np.array(backbone_out.getLayerInt32("mask_z"))
            .reshape((1, 64))
            .astype(np.bool8)
        )

        self.pos_z = np.array(backbone_out.getLayerFp16("pos_z")).reshape((64, 1, 128))

        self.state = bbox

        self.fps.tick()
        self.logger.info("Initialization done...")

        return input_frame, bbox

    def track(
        self, input_frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        H, W, _ = input_frame.shape
        img_x, resize_factor = self.sample_target(
            input_frame, self.state, self.SEARCH_FACTOR, self.SEARCH_SIZE
        )

        img_x_buff = dai.Buffer()
        img_x_buff.setData(
            img_x.transpose(2, 0, 1).flatten().astype(np.uint8).view(np.uint8)
        )

        feat_z_buff = dai.Buffer()
        feat_z_buff.setData(self.feat_z.flatten().astype(np.float16).view(np.uint8))

        mask_z_buff = dai.Buffer()
        mask_z_buff.setData(self.mask_z.flatten().view(np.uint8))

        pos_z_buff = dai.Buffer()
        pos_z_buff.setData(self.pos_z.flatten().astype(np.float16).view(np.uint8))

        self.q_in_complete_img_x.send(img_x_buff)
        self.q_in_complete_feat_z.send(feat_z_buff)
        self.q_in_complete_mask_z.send(mask_z_buff)
        self.q_in_complete_pos_z.send(pos_z_buff)

        complete_out = self.q_out_complete.get()

        outputs_coord = np.array(complete_out.getLayerFp16("outputs_coord")).reshape(4)
        pred_box = outputs_coord * self.SEARCH_SIZE / resize_factor
        self.state = self.clip_bbox(
            self.map_box_back(pred_box, self.state, resize_factor, self.SEARCH_SIZE),
            H,
            W,
            margin=10,
        )

        self.fps.tick()
        fps, avg_fps = self.fps.get()
        self.logger.info(f"Avg FPS: {avg_fps}  |  FPS: {fps}  |  bbox: {self.state}")

        return input_frame, self.state


class HostModeCamTracker(HostModeTracker):
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
        pre_backbone_nn_path: str,
        pre_complete_nn_path: str,
    ) -> None:
        super().__init__(
            backbone_nn_path,
            complete_nn_path,
            pre_backbone_nn_path,
            pre_complete_nn_path,
            TrackerType.CAM,
        )

        self.ix = 0
        self.iy = 0
        self.iw = 0
        self.ih = 0
        self.mouse_pressed = False
        self.initialize_tracker = False
        self.initialized = False

    def new_bbox_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.ix, self.iy = x, y
            self.initialized = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                self.iw, self.ih = x - self.ix, y - self.iy
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            self.initialize_tracker = True
            self.logger.info(f"New bbox: {self.ix, self.iy, self.iw, self.ih}")
            if self.iw < 0:
                self.ix += self.iw
                self.iw = abs(self.iw)
            if self.ih < 0:
                self.iy += self.ih
                self.ih = abs(self.ih)
            self.logger.info(f"New bbox: {self.ix, self.iy, self.iw, self.ih}")

    def run(self):
        if self.initialize_tracker and not self.initialized:
            self.initialize_tracker = False
            self.initialized = True
            out_img, bbox = self.initialize((self.ix, self.iy, self.iw, self.ih))
        elif self.initialized:
            out_img, bbox = self.track()
        else:
            out_img = self.q_out_img.get().getCvFrame()
            bbox = [0, 0, out_img.shape[1], out_img.shape[0]]

        return out_img, bbox

    def initialize(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        self.fps.reset()
        self.fps.start()

        input_frame = self.q_out_img.get().getCvFrame()

        img, _ = self.sample_target(
            input_frame, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE
        )

        img_buff = dai.Buffer()
        img_buff.setData(
            img.transpose(2, 0, 1).flatten().astype(np.uint8).view(np.uint8)
        )

        self.q_in_backbone_img.send(img_buff)

        backbone_out = self.q_out_backbone.get()

        self.feat_z = (
            np.array(backbone_out.getLayerFp16("feat_z"))
            .reshape((64, 1, 128))
            .astype(np.float32)
        )

        self.mask_z = (
            np.array(backbone_out.getLayerInt32("mask_z"))
            .reshape((1, 64))
            .astype(np.bool8)
        )

        self.pos_z = np.array(backbone_out.getLayerFp16("pos_z")).reshape((64, 1, 128))

        self.state = bbox

        self.fps.tick()

        return input_frame, bbox

    def track(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        timer_start = time.time() * 1000

        input_frame = self.q_out_img.get().getCvFrame()

        H, W, _ = input_frame.shape
        img_x, resize_factor = self.sample_target(
            input_frame, self.state, self.SEARCH_FACTOR, self.SEARCH_SIZE
        )

        img_x_buff = dai.Buffer()
        img_x_buff.setData(
            img_x.transpose(2, 0, 1).flatten().astype(np.uint8).view(np.uint8)
        )

        feat_z_buff = dai.Buffer()
        feat_z_buff.setData(self.feat_z.flatten().astype(np.float16).view(np.uint8))

        mask_z_buff = dai.Buffer()
        mask_z_buff.setData(self.mask_z.flatten().view(np.uint8))

        pos_z_buff = dai.Buffer()
        pos_z_buff.setData(self.pos_z.flatten().astype(np.float16).view(np.uint8))

        self.q_in_complete_img_x.send(img_x_buff)
        self.q_in_complete_feat_z.send(feat_z_buff)
        self.q_in_complete_mask_z.send(mask_z_buff)
        self.q_in_complete_pos_z.send(pos_z_buff)

        time_start = time.time() * 1000

        complete_out = self.q_out_complete.get()

        print("end: ", (time.time() * 1000) - time_start)

        outputs_coord = np.array(complete_out.getLayerFp16("outputs_coord")).reshape(4)
        pred_box = outputs_coord * self.SEARCH_SIZE / resize_factor
        self.state = self.clip_bbox(
            self.map_box_back(pred_box, self.state, resize_factor, self.SEARCH_SIZE),
            H,
            W,
            margin=10,
        )

        self.fps.tick()
        fps, avg_fps = self.fps.get()
        self.logger.info(f"Avg FPS: {avg_fps}  |  FPS: {fps}  |  bbox: {self.state}")
        self.logger.info(f"{time.time() * 1000 - timer_start}")

        return input_frame, self.state
