from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16
from keras import layers, models
from keras.src.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.src.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.src.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.src.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.src.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

INPUT_FOLDER = ""
GRAPH = True


TRANSFER = True
SIZE = (224,224)
EPOCHS = 20
BATCH_SIZE = 32

MODELS_DICT_TRANSFER = {
    "MobileNet": (MobileNetV2, mobilenet_v2_preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess_input),
    "ResNet50": (ResNet50, resnet_preprocess_input),
    "InceptionV3": (InceptionV3, inception_v3_preprocess_input),
    "VGG16": (VGG16, vgg16_preprocess_input),
}

MODELS_DICT_SEQUENTIAL = {
    "sequential": (None, layers.Rescaling(1. / 255))
}

SIGN_NAME = ['A-1', 'A-11a', 'A-16', 'A-17', 'A-2', 'A-20', 'A-21', 'A-29', 'A-30', 'A-6a', 'A-6b',
               'A-6c', 'A-7', 'B-1', 'B-2', 'B-20', 'B-21', 'B-22', 'B-23', 'B-25', 'B-33', 'B-34',
               'B-36', 'B-43', 'B-44', 'B-5', 'B-9', 'C-10', 'C-12', 'C-2', 'C-5', 'C-9', 'D-1',
               'D-15', 'D-18', 'D-2', 'D-23', 'D-3', 'D-42', 'D-43', 'D-4a', 'D-51', 'D-6', 'D-6b']

CLASS_NAMES = {
    0: "Niebezpieczny zakret w prawo",
    1: "Prog zwalniajacy",
    2: "Przejscie dla pieszych",
    3: "Dzieci",
    4: "Niebezpieczny zakret w lewo",
    5: "Odcinek jezdni o ruchu dwukierunkowym",
    6: "Tramwaj",
    7: "Sygnaly swietlne",
    8: "Inne niebezpieczenstwo",
    9: "Skrzyzowanie z droga podporzadkowana wystepujaca po obu stronach",
    10: "Skrzyzowanie z droga podporzadkowana wystepujaca po prawej stronie",
    11: "Skrzyzowanie z droga podporzadkowana wystepujaca po lewej stronie",
    12: "Ustap pierwszenstwa",

    13: "Zakaz ruchu w obu kierunkach",
    14: "Zakaz wjazdu",
    15: "Stop",
    16: "Zakaz skretu w lewo i zawracania",
    17: "Zakaz skretu w prawo i zawracania",
    18: "Zakaz zawracania",
    19: "Zakaz wyprzedzania",
    20: "Ograniczenie predkosci",
    21: "Koniec ograniczenia predkosci",
    22: "Zakaz zatrzymywania sie",
    23: "Strefa ograniczonej predkosci",
    24: "Koniec strefy ograniczonej predkosci",
    25: "Zakaz wjazdu samochodów ciezarowych",
    26: "Zakaz wjazdu rowerow",

    27: "Nakaz jazdy z lewej strony znaku",
    28: "Ruch okrezny",
    29: "Nakaz jazdy w prawo za znakiem",
    30: "Nakaz jazdy prosto",
    31: "Nakaz jazdy z prawej strony znaku",

    32: "Droga z pierwszenstwem",
    33: "Przystanek autobusowy",
    34: "Parking",
    35: "Koniec drogi z pierwszenstwem",
    36: "Stacja paliwowa",
    37: "Droga jednokierunkowa",
    38: "Obszar zabudowany",
    39: "Koniec obszaru zabudowanego",
    40: "Droga bez przejazdu",
    41: "Automatyczna kontrola predkości",
    42: "Przejscie dla pieszych",
    43: "Przejscie dla pieszych i przejazd dla rowerzystow",
}