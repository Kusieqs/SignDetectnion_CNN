from ultralytics import YOLO
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from constants import CLASS_NAMES, SIZE

yolo_model = YOLO("runs/detect/train6/weights/best.pt")
classifier_model = tf.keras.models.load_model("0.994.keras")


image_path = r"test.png"
image = cv2.imread(image_path)

results = yolo_model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_sign = image[y1:y2, x1:x2]


        sign = cv2.resize(cropped_sign, (224,224))
        cv2.imshow("sign", sign)
        cv2.waitKey(0)


        normalization_sign = np.array(sign) / 255.0
        img_array = np.expand_dims(normalization_sign, axis=0)


        prediction = classifier_model.predict(img_array)
        class_id = np.argmax(prediction)
        label = CLASS_NAMES.get(class_id, "Unknown")



        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({prediction[0][class_id]:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 7️⃣ Zapisz lub pokaż obraz z wynikami
cv2.imshow("Detection & Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()