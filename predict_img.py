import os
import numpy as np
import cv2
import tensorflow as tf
from Utils.constants import MODELS_DICT_TRANSFER, MODELS_DICT_SEQUENTIAL, SIZE, CLASS_NAMES
from ultralytics import YOLO

def classify_batch(crops):
    if not crops:
        return []

    if MODEL_NAME == "sequential":
        preprocessed = [MODELS_DICT_TRANSFER.get(MODEL_NAME)[1](cv2.resize(crop, SIZE)) for crop in crops]
    else:
        preprocessed = [MODELS_DICT_SEQUENTIAL.get(MODEL_NAME)[1](cv2.resize(crop, SIZE)) for crop in crops]

    batch_array = np.array(preprocessed)

    predictions = classifier_model.predict(batch_array, verbose=0)

    results = [(np.argmax(pred), pred) for pred in predictions]
    return results


YOLO_MODEL = YOLO("MainModels/best.pt")
classifier_model = tf.keras.models.load_model("MainModels/0.991.keras")
MODEL_NAME = "VGG16"


for file in os.listdir('TestPict'):
    path = os.path.join('TestPict', file)
    image = cv2.imread(path)
    image = cv2.resize(image, (1100, 750))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = YOLO_MODEL(image, verbose=False)

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

    count = 0
    for (x1, y1, x2, y2), label, confidence in shape_and_name:
        count += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{count}. {label} ({confidence:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"{count}. {label} ({confidence:.1f}%)")

    print("###\n\n\n\n\n\n\n###")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()