import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_dirs(dir):
    for split in ['train', 'val']:
        path = os.path.join(dir, split)
        os.makedirs(path, exist_ok=True)

def ensure_class_dirs(splitdir, name):
    for split in ['train', 'val']:
        class_dir = os.path.join(splitdir, split, name)
        os.makedirs(class_dir, exist_ok=True)

def split_data(source, split):
    create_dirs(split)

    for name in os.listdir(source):
        path = os.path.join(source, name)

        if not os.path.isdir(path):
            continue

        images = os.listdir(path)
        random.shuffle(images)

        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        print(len(images), len(train_images), len(val_images))

        ensure_class_dirs(split, name)

        for img_name in train_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'train', name, img_name))

        for img_name in val_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'val', name, img_name))
