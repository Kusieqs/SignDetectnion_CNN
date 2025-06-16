from ultralytics import YOLO
from constants import *
model = YOLO("yolov8n.pt")
model.train(data=YAML_PATH, epochs=EPOCHS, imgsz=SIZE, batch=BATCH_SIZE)