import os
from utils import data_split
from utils.preparation_data import apply_and_save_augmentations
from utils.create_dirs import create_dirs
from model import  compile_model

if __name__ == "__main__":
    input_folder = "BasicSigns\classification"
    augmentations_folder = "AugmentationSigns"
    final_folder = 'DataSplit'

    create_dirs()

    if len(os.listdir(augmentations_folder)) == 0:
        print("Augmentation data...")
        apply_and_save_augmentations(input_folder, augmentations_folder)

    if len(os.listdir(final_folder)) == 0:
        print("Spliting data...")
        data_split.split_data(augmentations_folder, final_folder)

    compile_model(final_folder)




