from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)

    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format
