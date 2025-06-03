import os
import random
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from Utils.constants import SIZE

def create_dirs(base_dir):
    for split in ['train', 'val']:
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)

def ensure_class_dirs(base_dir, class_name):
    for split in ['train', 'val']:
        class_path = os.path.join(base_dir, split, class_name)
        os.makedirs(class_path, exist_ok=True)

def resize_image(image_path, size):
    img = cv2.imread(image_path)
    if img is not None:
        resized_img = cv2.resize(img, size)
        return resized_img
    else:
        print(f"Błąd podczas wczytywania obrazu: {image_path}")
        return None

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def translate_image(image, x, y):
    (h, w) = image.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_and_save_augmentations(image, base_path, img_name):
    cv2.imwrite(os.path.join(base_path, img_name), image)
    cv2.imwrite(os.path.join(base_path, 'a_'+img_name), rotate_image(image, 5))
    cv2.imwrite(os.path.join(base_path, 'b_'+img_name), rotate_image(image, -5))
    cv2.imwrite(os.path.join(base_path, 'c_'+img_name), translate_image(image, 10, 10))
    cv2.imwrite(os.path.join(base_path, 'd_'+img_name), translate_image(image, -10, -10))

def split_and_augment_data(source_folder, final_folder):
    create_dirs(final_folder)

    for class_name in os.listdir(source_folder):
        print(class_name)
        class_path = os.path.join(source_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        ensure_class_dirs(final_folder, class_name)

        images = os.listdir(class_path)
        random.shuffle(images)
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for img_name in train_imgs:
            img_path = os.path.join(class_path, img_name)
            image = resize_image(img_path, SIZE)
            if image is not None:
                save_path = os.path.join(final_folder, 'train', class_name)
                apply_and_save_augmentations(image, save_path, img_name)

        for img_name in val_imgs:
            img_path = os.path.join(class_path, img_name)
            image = resize_image(img_path, SIZE)
            if image is not None:
                save_path = os.path.join(final_folder, 'val', class_name)
                apply_and_save_augmentations(image, save_path, img_name)