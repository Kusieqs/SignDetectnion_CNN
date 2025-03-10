import os

from utils import data_split
from utils.preparation_data import apply_and_save_augmentations
from model import  compile_model

if __name__ == "__main__":
    input_folder = "Znaki\classification"
    output_folder = "MainSigns"

    if len(os.listdir(output_folder)) == 0:
        print("Augmentation data...")
        apply_and_save_augmentations(input_folder, output_folder)

    source_dir = 'MainSigns'
    data_dir = 'DataSplit'

    if len(os.listdir(source_dir)) == 0:
        print("Spliting data...")
        data_split.split_data(source_dir, data_dir)

    compile_model(data_dir)




