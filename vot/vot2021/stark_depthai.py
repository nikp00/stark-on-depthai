import vot
import cv2
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, base_path.as_posix())
from tracker import HostModeSyntheticTracker

if __name__ == "__main__":
    tracker = HostModeSyntheticTracker(
        base_path.joinpath(
            "models/backbone_bottleneck_pe/backbone_bottleneck_pe_myriad.blob"
        ).as_posix(),
        base_path.joinpath("models/complete/complete_myriad_sim.blob").as_posix(),
        base_path.joinpath("models/to_float_model/to_float_model_128.blob").as_posix(),
        base_path.joinpath("models/to_float_model/to_float_model_320.blob").as_posix(),
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

            state = [left, top, width, height]
            tracker.initialize(img, state)

        else:
            img, state = tracker.track(img)
            print("BBOX after tracking: ", state)
            handle.report(vot.Rectangle(*state))

        i += 1

    tracker.device.close()
