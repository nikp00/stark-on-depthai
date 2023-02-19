import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class PreModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_img: Tensor) -> Tensor:
        return in_img.to(torch.float)


if __name__ == "__main__":
    config = [
        {"resolution": 128, "output_name": "img"},
        {"resolution": 320, "output_name": "img_x"},
    ]
    for cfg in config:
        save_name = f"to_float_model_{cfg['resolution']}.onnx"
        img_z = torch.zeros(
            (cfg["resolution"], cfg["resolution"], 3), dtype=torch.uint8
        )
        # mask_z = torch.zeros((resolution, resolution))

        torch_model = PreModel()

        print("Converting preprocessing model to ONNX...")

        torch.onnx.export(
            torch_model,  # model being run
            (img_z),  # model input (or a tuple for multiple inputs)
            save_name,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["in_img"],  # the model's input names
            output_names=[cfg["output_name"]],  # the model's output names
        )

    print("Done!")
