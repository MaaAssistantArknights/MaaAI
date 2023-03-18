from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="datasets/label.yaml", epochs=1000, batch=128)  # train the model
