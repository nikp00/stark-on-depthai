import onnx
import onnxruntime
import cv2
import numpy as np
import math

from typing import Union, Tuple


class OnnxTracker:
    TEMPLATE_SIZE = 128
    TEMPLATE_FACTOR = 2.0
    SEARCH_SIZE = 320
    SEARCH_FACTOR = 5.0

    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
    ) -> None:
        self.backbone = onnxruntime.InferenceSession(backbone_nn_path)
        self.complete = onnxruntime.InferenceSession(complete_nn_path)

        self.feat_z = None
        self.mask_z = None
        self.pos_z = None
        self.state = None

    def sample_target(
        self,
        im: np.ndarray,
        bbox: list,
        search_area_factor: float,
        output_sz: Union[int, float],
    ):
        if not isinstance(bbox, list):
            x, y, w, h = bbox.tolist()
        else:
            x, y, w, h = bbox
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

    def map_box_back(
        self,
        pred_box: np.ndarray,
        state: np.ndarray,
        resize_factor: float,
        search_size: float,
    ) -> np.ndarray:
        cx_prev, cy_prev = (
            state[0] + 0.5 * state[2],
            state[1] + 0.5 * state[3],
        )
        cx, cy, w, h = pred_box
        half_side = 0.5 * search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return np.array([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h])

    def clip_box(self, box: np.ndarray, H: int, W: int, margin: int = 0) -> np.ndarray:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return np.array([x1, y1, w, h])

    def initialize(
        self, input_frame: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        img, _ = self.sample_target(
            input_frame, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE
        )

        img = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img = img.astype(np.float32)

        ort_inputs = {"img": img}

        ort_outs = self.backbone.run(None, ort_inputs)

        self.feat_z = ort_outs[0]
        self.mask_z = ort_outs[1]
        self.pos_z = ort_outs[2]
        self.state = np.array(bbox)

        return input_frame, self.state.tolist()

    def track(
        self,
        input_frame: np.ndarray,
    ):
        H, W, _ = input_frame.shape
        img_x, resize_factor = self.sample_target(
            input_frame, self.state, self.SEARCH_FACTOR, self.SEARCH_SIZE
        )  # (x1, y1, w, h)

        img_x = img_x[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_x = img_x.astype(np.float32)

        print(self.feat_z.dtype)
        ort_inputs = {
            "img_x": img_x,
            "feat_z": self.feat_z,
            "mask_z": self.mask_z,
            "pos_z": self.pos_z,
        }

        ort_outs = self.complete.run(None, ort_inputs)

        outputs_coord = ort_outs[0][0]

        print(ort_outs)

        pred_box = outputs_coord * self.SEARCH_SIZE / resize_factor
        self.state = self.clip_box(
            self.map_box_back(pred_box, self.state, resize_factor, self.SEARCH_SIZE),
            H,
            W,
            margin=10,
        )

        return input_frame, self.state.tolist()
