#!/usr/bin/env python

import numpy as np
import vot
import cv2
import os


from tracker import OnnxTracker

from typing import List, Tuple, Optional, Union


class Tracker:
    def __init__(
        self,
        backbone_nn_path: str,
        complete_nn_path: str,
    ) -> None:
        self.template_size = 128
        self.template_factor = 2.0
        self.search_size = 320
        self.search_factor = 5.0

        self.backbone = onnxruntime.InferenceSession(backbone_nn_path)
        self.complete = onnxruntime.InferenceSession(complete_nn_path)

        self.feat_z = None
        self.mask_z = None
        self.pos_z = None
        self.state = None

        self.handle = vot.VOT("rectangle")
        self.selection = self.handle.region()
        self.state = []

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

    def run(self):
        i = 0
        while imgfile := self.handle.frame():
            img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
            if i == 0:
                top = max(self.selection.y, 0)
                left = max(self.selection.x, 0)
                width = self.selection.width
                height = self.selection.height

                self.state = np.array([left, top, width, height])

                self.feat_z, self.pos_z, self.mask_z, self.state = self.initialize(
                    img, self.state
                )
            else:
                self.state = self.track(
                    img, self.feat_z, self.pos_z, self.mask_z, self.state
                )

                x1, y1, w, h = self.state.tolist()
                out = cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x1 + w), int(y1 + h)),
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.imwrite(
                    f"/home/nik/Projects/Diplomska/stark-on-depthai/vot/debug/{i}.jpg",
                    out,
                )

                self.handle.report(vot.Rectangle(*self.state.tolist()))

            i += 1

    def initialize(
        self, img: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img, _ = self.sample_target(img, bbox, self.template_factor, self.template_size)

        ort_inputs = {
            # "img": img[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float),
            "img": torch.from_numpy(img.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(torch.float)
            .numpy(),
        }

        ort_outs = self.backbone.run(None, ort_inputs)

        feat_z = torch.from_numpy(ort_outs[0]).to(torch.float).numpy()
        mask_z = torch.from_numpy(ort_outs[1]).to(torch.bool).numpy()
        pos_z = torch.from_numpy(ort_outs[2]).to(torch.float).numpy()
        state = np.array(bbox)

        return feat_z, pos_z, mask_z, state

    def track(
        self,
        img: np.ndarray,
        feat_z: np.ndarray,
        pos_z: np.ndarray,
        mask_z: np.ndarray,
        state: np.ndarray,
    ):
        H, W, _ = img.shape
        img_x, resize_factor = self.sample_target(
            img, state, self.search_factor, self.search_size
        )  # (x1, y1, w, h)

        ort_inputs = {
            "img_x": torch.from_numpy(img_x.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(torch.float)
            .numpy(),
            "feat_z": feat_z,
            "mask_z": mask_z,
            "pos_z": pos_z,
        }

        ort_outs = self.complete.run(None, ort_inputs)

        outputs_coord = ort_outs[0][0]

        pred_box = outputs_coord * self.search_size / resize_factor
        state = self.clip_box(
            self.map_box_back(pred_box, state, resize_factor, self.search_size),
            H,
            W,
            margin=10,
        )

        return state


if __name__ == "__main__":
    tracker = OnnxTracker(
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe.onnx",
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/complete/complete.onnx",
    )

    handle = vot.VOT("rectangle")
    selection = handle.region()
    state = []

    i = 0
    while imgfile := handle.frame():
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        if i == 0:
            top = max(selection.y, 0)
            left = max(selection.x, 0)
            width = selection.width
            height = selection.height

            state = np.array([left, top, width, height])

            _, state = tracker.initialize(img, state)

        else:
            _, state = tracker.track(img)

            handle.report(vot.Rectangle(*state))

            x1, y1, w, h = state
            out = cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                color=(0, 0, 255),
                thickness=2,
            )

            debug_path = imgfile.split("/")[-3:]
            debug_path.pop(1)
            base_path = f"/home/nik/Projects/Diplomska/stark-on-depthai/vot/debug/Stark_onnx/{debug_path[0]}"
            os.makedirs(base_path, exist_ok=True)

            cv2.imwrite(f"{base_path}/{debug_path[1]}", out)

        i += 1
