import data_split
from preparation_data import apply_and_save_augmentations
from data_split import split_data
import tensorflow as tf

if __name__ == "__main__":
    input_folder = r"C:\Users\konra\PycharmProjects\CNN_SD\Znaki\classification"
    output_folder = r"C:\Users\konra\PycharmProjects\CNN_SD\MainSigns"

    apply_and_save_augmentations(input_folder, output_folder)

    Source_dir = r'C:\Users\konra\PycharmProjects\CNN_SD\MainSigns'
    Split_dir = r'C:\Users\konra\PycharmProjects\CNN_SD\DataSplit'

    data_split.split_data(Source_dir, Split_dir)




