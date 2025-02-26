import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_dirs(dir):
    for split in ['train', 'test', 'val']:
        path = os.path.join(dir, split)
        os.makedirs(path, exist_ok=True)

def ensure_class_dirs(splitdir, name):
    for split in ['train', 'test', 'val']:
        class_dir = os.path.join(splitdir, split, name)
        os.makedirs(class_dir, exist_ok=True)

def split_data(source, split):

    create_dirs(split)

    #train = 0.7
    #test = 0.15
    #val = 0.15

    for name in os.listdir(source):
        path = os.path.join(source, name)

        if not os.path.isdir(path):
            continue

        images = os.listdir(path)
        random.shuffle(images)

        train_images, test_val_images = train_test_split(images, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(test_val_images, test_size=0.5,
                                                   random_state=42)
        print(len(images), len(train_images), len(val_images), len(test_images))

        ensure_class_dirs(split, name)

        for img_name in train_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'train', name, img_name))

        for img_name in val_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'val', name, img_name))

        for img_name in test_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'test', name, img_name))


