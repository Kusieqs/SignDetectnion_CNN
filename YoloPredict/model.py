from ultralytics import YOLO
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")

model.train(data=r"C:\Users\konra\PycharmProjects\CNN_SD\YoloPredict\data.yaml", epochs=20, imgsz=640, batch=32)