import time

from ultralytics import YOLO
import cv2
import tensorflow as tf
import numpy as np
from utils.constants import CLASS_NAMES, SIZE, MODELS_DICT

yolo_model = YOLO("runs/detect/train6/weights/best.pt")
classifier_model = tf.keras.models.load_model("0.942.keras")


video_path = r"1.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
skip_frames = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    frame = cv2.resize(frame, (640, 416))
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_sign = frame[y1:y2, x1:x2]

            if cropped_sign.size == 0:
                continue

            sign = cv2.resize(cropped_sign, SIZE)
            method = MODELS_DICT.get("EfficientNetB0")[1]
            img_preprocessed = method(sign)
            img_array = np.expand_dims(img_preprocessed, axis=0)

            prediction = classifier_model.predict(img_array)
            class_id = np.argmax(prediction)
            label = CLASS_NAMES.get(class_id, "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({prediction[0][class_id]:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
