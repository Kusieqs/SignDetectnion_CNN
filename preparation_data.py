import cv2
import os
import numpy as np

from constants import SIZE


def resize_image(image_path, size):
    img = cv2.imread(image_path)
    if img is not None:
        resized_img = cv2.resize(img, size)
        return resized_img
    else:
        print(f"Błąd podczas wczytywania obrazu: {image_path}")
        return None


def apply_and_save_augmentations(folder_path, output_folder):

    for folder in os.listdir(folder_path):
        print(f"Wczytanie folderu: {folder}")
        path = os.path.join(folder_path, folder)
        path_to_MainSigns = os.path.join(output_folder, folder)

        if not os.path.exists(path_to_MainSigns):
            os.makedirs(path_to_MainSigns)

        for images in os.listdir(path):
            path_to_image = os.path.join(path, images)
            image = resize_image(path_to_image, SIZE)

            rotated_img1 = rotate_image(image,5)
            rotated_img2 = rotate_image(rotated_img1,-10)
            translate_image1 = translate_image(rotated_img1,10, 10)
            translate_image2 = translate_image(rotated_img2,-10, -10)

            cv2.imwrite(os.path.join(path_to_MainSigns, images), image)
            cv2.imwrite(os.path.join(path_to_MainSigns,'a'+images), rotated_img1)
            cv2.imwrite(os.path.join(path_to_MainSigns,'b'+images), rotated_img2)
            cv2.imwrite(os.path.join(path_to_MainSigns,'c'+images), translate_image1)
            cv2.imwrite(os.path.join(path_to_MainSigns,'d'+images), translate_image2)
            # Jasnosc?


# rotacja obrazu o stopnie
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# przesuniecie obrazu
def translate_image(image, x, y):
    (h, w) = image.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated





