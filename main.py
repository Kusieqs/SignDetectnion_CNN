from utils import data_split
from utils.preparation_data import apply_and_save_augmentations
from model import  compile_model

if __name__ == "__main__":
    input_folder = r"C:\Users\konra\PycharmProjects\CNN_SD\Znaki\classification"
    output_folder = r"C:\Users\konra\PycharmProjects\CNN_SD\MainSigns"

    apply_and_save_augmentations(input_folder, output_folder)

    source_dir = r'C:\Users\konra\PycharmProjects\CNN_SD\MainSigns'
    data_dir = r'C:\Users\konra\PycharmProjects\CNN_SD\DataSplit'

    data_split.split_data(source_dir, data_dir)
    compile_model(data_dir)




