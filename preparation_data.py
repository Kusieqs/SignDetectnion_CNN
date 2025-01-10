
import cv2
import os

def resize_image(image_path, size):
    img = cv2.imread(image_path)
    if img is not None:
        resized_img = cv2.resize(img, (size, size))
        return resized_img
    else:
        print(f"Błąd podczas wczytywania obrazu: {image_path}")
        return None

def resize_images_in_folder(folder_path, output_folder):
    size = 200

    for folder in os.listdir(folder_path):
        print(f"Wczytanie folderu: {folder}")
        path = os.path.join(folder_path, folder)
        path_to_MainSigns = os.path.join(r"C:\Users\konra\PycharmProjects\CNN_SD\MainSigns", folder)

        if not os.path.exists(path_to_MainSigns):
            os.makedirs(path_to_MainSigns)

        for images in os.listdir(path):
            path_to_image = os.path.join(path, images)
            image = resize_image(path_to_image, size)

            cv2.imwrite(os.path.join(path_to_MainSigns, images), image)

