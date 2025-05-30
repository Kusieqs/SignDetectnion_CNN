import os
import cv2
from Utils import data_split
from Utils.constants import TRANSFER, SIZE, INPUT_FOLDER
from Utils.preparation_data import apply_and_save_augmentations
from Utils.create_dirs import create_dirs
from transfer_learning import compile_model_transfer_learning
from model_CNN import compile_model

if __name__ == "__main__":
    augmentations_folder = "AugmentationSigns"
    final_folder = 'DataSplit'

    create_dirs()

    if len(os.listdir(augmentations_folder)) == 0:
        print("Augmentation data...")
        apply_and_save_augmentations(INPUT_FOLDER, augmentations_folder)

    if len(os.listdir(final_folder)) == 0:
        print("Spliting data...")
        data_split.split_data(augmentations_folder, final_folder)
    else:
        for folder in os.listdir(final_folder):
            for signClass in os.listdir(os.path.join(final_folder, folder)):
                for file in os.listdir(os.path.join(final_folder, folder, signClass)):
                    path = os.path.join(final_folder, folder, signClass, file)
                    img = cv2.imread(path)
                    resized_img = cv2.resize(img, SIZE)
                    cv2.imwrite(path, resized_img)

    if TRANSFER:
        compile_model_transfer_learning(final_folder)
    else:
        compile_model(final_folder)