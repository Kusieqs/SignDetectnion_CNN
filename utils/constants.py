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

SIZE = (224,224)
EPOCHS = 20
BATCH_SIZE = 32


MODELS_DICT = {
    "ResNet50": (ResNet50, resnet_preprocess_input),
    "InceptionV3": (InceptionV3, inception_v3_preprocess_input),
    "MobileNet": (MobileNetV2, mobilenet_v2_preprocess_input),
    "VGG16": (VGG16, vgg16_preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess_input)
}


CLASS_NAMES = {
    0: "Niebezpieczny zakret w prawo",
    1: "Prog zwalniajacy",
    2: "Roboty drogowe",
    3: "Przejscie dla pieszych",
    4: "Dzieci",
    5: "Niebezpieczny zakret w lewo",
    6: "Odcinek jezdni o ruchu dwukierunkowym",
    7: "Tramwaj",
    8: "Sygnaly swietlne",
    9: "Inne niebezpieczenstwo",
    10: "Skrzyzowanie z drogą podporzadkowana wystepujaca po obu stronach",
    11: "Skrzyzowanie z drogą podporzadkowana wystepujaca po prawej stronie",
    12: "Skrzyzowanie z drogą podporzadkowana wystepujaca po lewej stronie",
    13: "Ustap pierwszenstwa",

    14: "Zakaz ruchu w obu kierunkach",
    15: "Zakaz wjazdu",
    16: "Stop",
    17: "Zakaz skretu w lewo i zawracania",
    18: "Zakaz skretu w prawo i zawracania",
    19: "Zakaz zawracania",
    20: "Zakaz wyprzedzania",
    21: "Zakaz wyprzedzania przez samochody ciezarowe",
    22: "Koniec zakazu wyprzedzania",
    23: "Ograniczenie predkosci",
    24: "Koniec ograniczenia prędkości",
    25: "Zakaz zatrzymywania sie",
    26: "Zakaz ruchu pieszych",
    27: "Strefa ograniczonej prędkości",
    28: "Koniec strefy ograniczonej predkości",
    29: "Zakaz wjazdu samochodów ciezarowych",
    30: "Zakaz wjazdu rowerow",

    31: "Nakaz jazdy z lewej strony znaku",
    32: "Ruch okrezny",
    33: "Nakaz jazdy w prawo za znakiem",
    34: "Nakaz jazdy w lewo za znakiem",
    35: "Nakaz jazdy prosto",
    36: "Nakaz jazdy z prawej strony znaku",

    37: "Droga z pierwszenstwem",
    38: "Przystanek autobusowy",
    39: "Parking",
    40: "Koniec drogi z pierwszenstwem",
    41: "Stacja paliwowa",
    42: "Droga jednokierunkowa",
    43: "Obszar zabudowany",
    44: "Koniec obszaru zabudowanego",
    45: "Droga bez przejazdu",
    46: "Automatyczna kontrola predkości",
    47: "Przejscie dla pieszych",
    48: "Przejscie dla pieszych i przejazd dla rowerzystow",
    49: "Droga ekspresowa",
    50: "Koniec drogi ekspresowej",
    51: "Autostrada",

    52: "Inny",

    53: "Tabliczka wskazujaca, że przejscie dla pieszych jest szczegolnie uczeszczane przez dzieci",
}