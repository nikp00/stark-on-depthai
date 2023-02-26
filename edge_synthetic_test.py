#!/usr/bin/env python

import cv2
import numpy as np
from tracker import EdgeModeSyntheticTracker

from tracker.utils import DatasetType, ImageReader

from time import sleep


def draw_bbox(img, bbox, color=(0, 0, 255), thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(
        img,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        color=color,
        thickness=thickness,
    )


tracker = EdgeModeSyntheticTracker(
    "models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob",
    "models/complete/complete_myriad_sim.blob",
    "models/to_float_model/to_float_model_128.blob",
    "models/to_float_model/to_float_model_320.blob",
    "tracker/utils/script_node",
    False,
    3,
)

reader = ImageReader("dataset", DatasetType.LASOT)


for img, bbox, img_id, dir_id, category_id in reader.read():
    print(f"Processing image {img_id}")
    if img_id == 0:
        out_img, out_bbox = tracker.initialize(img, bbox)
    else:
        out_img, out_bbox = tracker.track(img)
        draw_bbox(out_img, out_bbox)

    cv2.imshow("Output img", out_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
