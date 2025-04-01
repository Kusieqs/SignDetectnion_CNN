from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data=r"C:\Users\konra\PycharmProjects\CNN_SD\YoloPredict\data.yaml", epochs=30, imgsz=(1100, 750), batch=32)