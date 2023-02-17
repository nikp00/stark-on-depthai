import numpy as np
import math
import cv2
import os
import time
from pathlib import Path

from typing import Optional, Callable, Tuple


class FpsCounter:
    def __init__(self) -> None:
        self.fps = 0
        self.start_time = 0
        self.counter = 0
        self.avg_fps = 0

    def start(self) -> None:
        self.start_time = time.monotonic()

    def tick(self) -> None:
        self.fps = 1 / (time.monotonic() - self.start_time)
        self.avg_fps += self.fps
        self.start_time = time.monotonic()
        self.counter += 1

    def stop(self) -> float:
        fps = self.get_fps()
        self.reset()
        return fps

    def reset(self) -> None:
        self.fps = 0
        self.counter = 0

    def get(self) -> float:
        return self.fps, self.avg_fps / max(self.counter, 1)


class DatasetType:
    GOT10K = "got10k/test"
    LASOT = "LASOT"


class ImageReader:
    def __init__(self, dataset_base_path: str, dataset: DatasetType):
        self.base_path = Path(dataset_base_path).absolute().resolve()
        self.dataset_path = Path(self.base_path, dataset).absolute()
        self.dataset = dataset

        if self.dataset == DatasetType.GOT10K:
            self.read_function = self.read_got10k
        elif self.dataset == DatasetType.LASOT:
            self.read_function = self.read_lasot

    def read(self):
        return self.read_function()

    def read_got10k(self):
        for dir_id, dir in enumerate(sorted(list(self.dataset_path.iterdir()))):
            if dir.is_dir():
                info = np.genfromtxt(
                    Path(dir, "groundtruth.txt"), delimiter=",", dtype=np.int32
                )
                images = sorted(list(Path(dir).glob("*.jpg")))
                img = cv2.imread(images[0].as_posix(), cv2.IMREAD_COLOR)

                yield img, info, 0, dir_id, 0

                for img_id, img in enumerate(images[1:-1]):
                    img = cv2.imread(img.as_posix(), cv2.IMREAD_COLOR)
                    yield img, info, img_id, dir_id, 0

    def read_lasot(self):
        for category_id, category in enumerate(
            sorted(list(self.dataset_path.iterdir()))
        ):
            for dir_id, dir in enumerate(sorted(list(category.iterdir()))):
                img_dir = Path(dir, "img")
                if dir.is_dir():
                    info = np.genfromtxt(
                        Path(dir, "groundtruth.txt"), delimiter=",", dtype=np.int32
                    )
                    images = sorted(list(Path(img_dir).glob("*.jpg")))
                    img = cv2.imread(images[0].as_posix(), cv2.IMREAD_COLOR)

                    yield img, info[0], 0, dir_id, category_id

                    for img_id, (img, info) in enumerate(zip(images[1:], info[1:]), 1):
                        img = cv2.imread(img.as_posix(), cv2.IMREAD_COLOR)
                        yield img, info, img_id, dir_id, category_id


class ResultWriter:
    def __init__(self, output_path: str, dataset_type: DatasetType) -> None:
        self.output_path = Path(output_path).absolute().resolve()
        self.dataset_type = dataset_type

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        img_id: int,
        dir_id: int,
        category_id: int = 0,
        ground_truth: Optional[np.ndarray] = None,
    ):
        if self.dataset_type == DatasetType.GOT10K:
            dir = Path(self.output_path, f"{dir_id:04d}")
            dir.mkdir(parents=True, exist_ok=True)
        elif self.dataset_type == DatasetType.LASOT:
            dir = Path(self.output_path, f"{category_id:03d}", f"{dir_id:03d}")
            dir.mkdir(parents=True, exist_ok=True)
            with open(Path(dir, "bbox.txt"), "a") as bbox_file:
                bbox_file.write(f"{','.join(map(str, map(int, bbox)))}\n")

            with open(Path(dir, "groundtruth.txt"), "a") as gt_file:
                gt_file.write(f"{','.join(map(str, map(int, ground_truth)))}\n")

        img_path = Path(f"{img_id:04d}.jpg")
        x1, y1, w, h = bbox
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x1 + w), int(y1 + h)),
            color=(0, 0, 255),
            thickness=2,
        )
        cv2.imwrite(Path(dir, img_path).as_posix(), img)
