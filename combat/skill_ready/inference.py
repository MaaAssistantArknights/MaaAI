from PIL import Image
from pathlib import Path
import numpy as np
import random

import onnxruntime as ort
import torchvision.transforms as transforms


def inference(path):
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    input_img = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()
    input_img = to_tensor(input_img)
    input_img = input_img.unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_img)}
    ort_output = ort_session.run(None, ort_inputs)[0][0]
    return ort_output


ort_session = ort.InferenceSession("checkpoints/best.onnx")

if __name__ == "__main__":
    rand_y = random.sample(list(Path("datasets/raw/y").glob("*")), 10)
    rand_n = random.sample(list(Path("datasets/raw/n").glob("*")), 10)
    for path in rand_y + rand_n:
        n_value, y_value = inference(path)
        out = "y" if y_value > n_value else "n"
        print(path, "\t", out)
