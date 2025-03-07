from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16

from keras.src.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.src.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.src.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.src.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.src.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

SIZE = (90,90)

MODELS_DICT = {
    "ResNet50": (ResNet50, resnet_preprocess_input),
    "InceptionV3": (InceptionV3, inception_v3_preprocess_input),
    "MobileNet": (MobileNetV2, mobilenet_v2_preprocess_input),
    "VGG16": (VGG16, vgg16_preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess_input)
}


CLASS_NAMES = {
    0: "Niebezpieczny zakret w prawo",
    1: "Niebezpieczny zakret w lewo",
    2: "Ustap pierwszenstwa",
    3: "Prog zwalniajacy",
    4: "Przejscie dla pieszych",
    5: "Dzieci",
    6: "Tramwaj",
    7: "Inne niebezpieczenstwo",
    8: "Zakaz ruchu w obu kierunkach",
    9: "Zakaz wjazdu",
    10: "Stop",
    11: "Zakaz skrecania w lewo",
    12: "Zakaz skrecania w prawo",
    13: "Zakaz zawracania",
    14: "Zakaz wyprzedzania",
    15: "Zakaz wyprzedzania przez samochody ciezarowe",
    16: "Ograniczenie predkosci",
    17: "Zakaz zatrzymywania sie",
    18: "Zakaz ruchu pieszych",
    19: "Nakaz jazdy w prawo za znakiem",
    20: "Nakaz jazdy w lewo za znakiem",
    21: "Nakaz jazdy prosto",
    22: "Nakaz jazdy z prawej strony znaku",
    23: "Ruch okrezny",
    24: "Droga z pierwszenstwem",
    25: "Droga jednokierunkowa",
    26: "Przejscie dla pieszych",
    27: "Przejscie dla pieszych i przejazd dla rowerzystow",
    28: "Parking",
    29: "Tabliczka wskazujaca, że przejscie dla pieszych jest szczegolnie uczeszczane przez dzieci",
}