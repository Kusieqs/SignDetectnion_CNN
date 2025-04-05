from ultralytics import YOLO
from constants import *
model = YOLO("yolov8n.pt")
model.train(data=r"C:\Users\konra\PycharmProjects\CNN_SD\YoloPredict\data.yaml", epochs=EPOCHS, imgsz=SIZE, batch=BATCH_SIZE)