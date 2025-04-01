import os

import numpy as np
from tensorflow.python.keras.saving.saved_model_experimental import sequential
from ultralytics import YOLO
import cv2
import tensorflow as tf
from Utils.constants import MODELS_DICT, SIZE, CLASS_NAMES

def classify_batch(crops):
    if not crops:
        return []

    preprocessed = [MODELS_DICT.get(model_name)[1](cv2.resize(crop, SIZE)) for crop in crops]
    batch_array = np.array(preprocessed)

    predictions = classifier_model.predict(batch_array, verbose=0)

    results = [(np.argmax(pred), pred) for pred in predictions]
    return results


yolo_model = YOLO("MainModels/best.pt")
classifier_model = tf.keras.models.load_model("MainModels/0.961.keras")
model_name = "MobileNet"
for file in os.listdir('TestPict'):
    path = os.path.join('TestPict', file)
    image = cv2.imread(path)
    image = cv2.resize(image, (1100, 750))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = yolo_model(image, verbose=False)

    shape_and_name = []
    crops = []
    positions = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_sign = image[y1:y2, x1:x2]
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
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()