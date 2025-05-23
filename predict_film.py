import cv2
import tensorflow as tf
import numpy as np
import threading
import queue
from ultralytics import YOLO
from Utils.constants import CLASS_NAMES, SIZE, MODELS_DICT_TRANSFER, MODELS_DICT_SEQUENTIAL

YOLO_MODEL = YOLO("")
CLASSIFIER_MODEL = tf.keras.models.load_model("")
MODEL_NAME = ""
VIDEO_PATH = ""

def classify_batch(crops):
    if not crops:
        return []

    if MODEL_NAME == "sequential":
        preprocessed = [MODELS_DICT_TRANSFER.get(MODEL_NAME)[1](cv2.resize(crop, SIZE)) for crop in crops]
    else:
        preprocessed = [MODELS_DICT_SEQUENTIAL.get(MODEL_NAME)[1](cv2.resize(crop, SIZE)) for crop in crops]

    batch_array = np.array(preprocessed)
    predictions = CLASSIFIER_MODEL.predict(batch_array, verbose=0)

    results = [(np.argmax(pred), pred) for pred in predictions]
    return results

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        results = YOLO_MODEL(frame, verbose=False)

        shape_and_name = []
        crops = []
        positions = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_sign = frame[y1:y2, x1:x2]
                if cropped_sign.size == 0:
                    continue
                crops.append(cropped_sign)
                positions.append((x1, y1, x2, y2))

        predictions = classify_batch(crops)

        for (x1, y1, x2, y2), (class_id, prediction) in zip(positions, predictions):
            label = CLASS_NAMES.get(class_id, "Unknown")
            confidence = prediction[class_id] * 100
            shape_and_name.append(((x1, y1, x2, y2), label, confidence))

        for (x1, y1, x2, y2), label, confidence in shape_and_name:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0
skip_frames = 1
frame_queue = queue.Queue(maxsize=5)

processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames == 0 and not frame_queue.full():
        frame = cv2.resize(frame, (1100, 750))
        frame_queue.put(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

frame_queue.put(None)
processing_thread.join()
cap.release()
cv2.destroyAllWindows()