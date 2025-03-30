from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data=r"C:\Users\konra\PycharmProjects\CNN_SD\YoloPredict\data.yaml", epochs=40, imgsz=(1200, 850), batch=32)