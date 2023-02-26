#!/usr/bin/env python

import cv2
import numpy as np
from tracker import EdgeModeCamTracker

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


tracker = EdgeModeCamTracker(
    "models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob",
    "models/complete/complete_myriad_sim.blob",
    "models/to_float_model/to_float_model_128.blob",
    "models/to_float_model/to_float_model_320.blob",
    "tracker/utils/script_node",
    True,
    3,
)

cv2.namedWindow("Output img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Output img", tracker.new_bbox_callback)

while True:
    out_img, bbox = tracker.run()

    draw_bbox(out_img, bbox)
    cv2.imshow("Output img", out_img)

    cmd = cv2.waitKey(1) & 0xFF
    if cmd == ord("q"):
        cv2.destroyAllWindows()
        break
