#!/usr/bin/env python

import vot
import cv2
import sys
from pathlib import Path

stark_path = Path(__file__).resolve().parent.parent.parent.joinpath("Stark")
sys.path.insert(0, stark_path.as_posix())
from tracking.stark_st import STARK_ST


if __name__ == "__main__":
    tracker = STARK_ST()

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

            info = {"init_bbox": [left, top, width, height]}
            tracker.initialize(img, info)
        else:
            out = tracker.track(img)
            state = out["target_bbox"]

            handle.report(vot.Rectangle(*state))

            x1, y1, w, h = state

        i += 1
