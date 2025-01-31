import os
import shutil
import random

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

    train = 0.7
    test = 0.15
    val = 0.15

    for name in os.listdir(source):
        path = os.path.join(source, name)

        if not os.path.isdir(path):
            continue

        images = os.listdir(path)
        random.shuffle(images)
        num_images = len(images)
        train_end = int(train * num_images)
        val_end = train_end + int(val * num_images)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        ensure_class_dirs(split, name)

        for img_name in train_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'train', name, img_name))

        for img_name in val_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'val', name, img_name))

        for img_name in test_images:
            shutil.copy(os.path.join(path, img_name), os.path.join(split, 'test', name, img_name))


