#!/usr/bin/env python

import numpy as np
import vot
import cv2
import sys
from pathlib import Path


base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, base_path.as_posix())
from tracker import OnnxTracker


if __name__ == "__main__":
    tracker = OnnxTracker(
        base_path.joinpath(
            "models/backbone_bottleneck_pe/backbone_bottleneck_pe.onnx"
        ).as_posix(),
        base_path.joinpath("models/complete/complete.onnx").as_posix(),
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

        i += 1
