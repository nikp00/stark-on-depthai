from tracker import HostModeSyntheticTracker
import vot
import cv2
import numpy as np
import os


def draw_bbox(img, bbox, color=(0, 0, 255), thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(
        img,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        color=color,
        thickness=thickness,
    )


if __name__ == "__main__":
    print("Deal")
    tracker = HostModeSyntheticTracker(
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob",
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/complete/complete_myriad_sim.blob",
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/to_float_model/to_float_model_128.blob",
        "/home/nik/Projects/Diplomska/stark-on-depthai/models/to_float_model/to_float_model_320.blob",
    )

    handle = vot.VOT("rectangle")
    selection = handle.region()
    state = []

    print("AAAAA")

    i = 0
    while imgfile := handle.frame():
        print(imgfile)
        img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
        print("New img...")
        if i == 0:
            top = max(selection.y, 0)
            left = max(selection.x, 0)
            width = selection.width
            height = selection.height

            state = [left, top, width, height]
            print("Initializing tracker...")
            tracker.initialize(img, state)

        else:
            img, state = tracker.track(img)
            print("BBOX after tracking: ", state)
            handle.report(vot.Rectangle(*state))

            draw_bbox(img, state)

            debug_path = imgfile.split("/")[-3:]
            debug_path.pop(1)
            base_path = f"/home/nik/Projects/Diplomska/stark-on-depthai/vot/debug/Stark/{debug_path[0]}"
            os.makedirs(base_path, exist_ok=True)

            cv2.imwrite(f"{base_path}/{debug_path[1]}", img)

        i += 1

    print("Closing device...")
    tracker.device.close()
    print("Device closed")
