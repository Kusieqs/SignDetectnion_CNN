import os
import cv2
from Utils.constants import TRANSFER, SIZE, IMAGES_PATH
from Utils.preparation_data import split_and_augment_data
from transfer_learning import compile_model_transfer_learning
from model_CNN import compile_model

if __name__ == "__main__":
    final_folder = 'DataSplit'

    if not os.path.exists(final_folder):
        os.makedirs(final_folder, exist_ok=True)

    if len(os.listdir(final_folder)) == 0:
        print("Wczytywanie i dzielenie danych:")
        split_and_augment_data(IMAGES_PATH, final_folder)
    else:
        for split in ['train', 'val']:
            for class_name in os.listdir(os.path.join(final_folder, split)):
                class_path = os.path.join(final_folder, split, class_name)
                for file in os.listdir(class_path):
                    path = os.path.join(class_path, file)
                    img = cv2.imread(path)
                    if img is not None:
                        resized_img = cv2.resize(img, SIZE)
                        cv2.imwrite(path, resized_img)

    if TRANSFER:
        compile_model_transfer_learning(final_folder)
    else:
        compile_model(final_folder)