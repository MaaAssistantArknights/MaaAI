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
    count = 0
    correct = 0
    for path in list(Path("datasets/test/y/").glob("*")):
        n_value, y_value = inference(path)
        out = True if y_value > n_value else False
        count += 1
        if out:
            correct += 1
        else:
            print('error', path)

    for path in  list(Path("datasets/test/n/").glob("*")):
        n_value, y_value = inference(path)
        out = True if y_value > n_value else False
        count += 1
        if not out:
            correct += 1
        else:
            print('error', path)

    print(f'{correct} / {count}, {correct / count}')