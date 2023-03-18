from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/train/weights/best.onnx")
    results = model("datasets/test/test.mp4", save=True)  # predict on an image
