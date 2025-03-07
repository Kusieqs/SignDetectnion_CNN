import os

import keras
from keras import layers, models
from keras.src.layers import GlobalAveragePooling2D
from keras.src.optimizers import Adam

from constants import SIZE, MODELS_DICT
from utils.generate_reports import generate_classification_report


def train_models(train_data, val_data, num_classes, img_size, epochs=20, batch_size=32):
    histories = {}

    preprocessed_datasets = {}
    for model_name, (_, preprocess_fn) in MODELS_DICT.items():
        preprocessed_datasets[model_name] = (
            train_data.map(lambda x, y: (preprocess_fn(x), y)),
            val_data.map(lambda x, y: (preprocess_fn(x), y))
        )

    for model_name, (fn_model, _) in MODELS_DICT.items():
        print(f"\nTraining model: {model_name} for img sizes: {img_size}x{img_size}")

        train_data_processed, val_data_processed = preprocessed_datasets[model_name]

        base_model = fn_model(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_data_processed,
                            validation_data=val_data_processed,
                            epochs=epochs,
                            batch_size=batch_size)
        histories[model_name] = (model, history, val_data_processed)

    return histories

def compile_model(data_dir):
    batch = 32
    data = data_dir

    train_ds = keras.utils.image_dataset_from_directory(
        f"{data}/train",
        image_size=SIZE,
        batch_size=batch,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=SIZE,
        batch_size=batch
    )

    class_names = ['A-1','A-2','A-7','A-11a','A-16','A-17','A-21','A-30','B-1','B-2','B-20','B-21',
                   'B-22','B-23','B-25','B-26','B-33','B-36','B-41','C-2','C-4','C-5','C-9','C-12',
                   'D-1','D-3','D-6','D-6b','D-18','T-27']

    models = train_models(train_ds, val_ds, len(class_names), SIZE[0])

    for model_name, (model, history, val_ds_processed) in models.items():
        print(f"Model: {model_name}")
        bal_acc = generate_classification_report(model, val_ds_processed, class_names,
                                                  model_name, SIZE[0])

        # Ścieżka do katalogu modelu dla danej kategorii

        model_path = f"{bal_acc:.3f}.keras"
        model.save(model_path)


    #normalization_layer = layers.Rescaling(1. / 255)
    #train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


