import os
import keras
from keras import layers, models
from keras.src.layers import GlobalAveragePooling2D
from keras.src.optimizers import Adam
import tensorflow as tf
from utils.constants import SIZE, MODELS_DICT
from utils.generate_reports import generate_classification_report

class_names = ['A-1', 'A-11a', 'A-14', 'A-16', 'A-17', 'A-2', 'A-20', 'A-21', 'A-29', 'A-30', 'A-6a', 'A-6b',
               'A-6c', 'A-7', 'B-1', 'B-2', 'B-20', 'B-21', 'B-22', 'B-23', 'B-25', 'B-26', 'B-27', 'B-33', 'B-34',
               'B-36', 'B-41', 'B-43', 'B-44', 'B-5', 'B-9', 'C-10', 'C-12', 'C-2', 'C-4', 'C-5', 'C-9', 'D-1',
               'D-15', 'D-18', 'D-2', 'D-23', 'D-3', 'D-42', 'D-43', 'D-4a', 'D-51', 'D-6', 'D-6b', 'D-7', 'D-8', 'D-9',
               'Inny']

def train_models(train_data, val_data, num_classes, img_size, epochs=20, batch_size=32):
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

        model.fit(train_data_processed,
                            validation_data=val_data_processed,
                            epochs=epochs,
                            batch_size=batch_size)


        path_to_save = os.path.join("models", model_name)
        os.makedirs("models", exist_ok=True)
        os.makedirs(path_to_save, exist_ok=True)

        bal_acc = generate_classification_report(model, val_data_processed, class_names,
                                                  model_name, SIZE[0], path_to_save)

        model_path = os.path.join(path_to_save, f"{bal_acc:.3f}.keras")
        model.save(model_path)


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

    print("Dostępne urządzenia GPU:", tf.config.list_physical_devices('GPU'))
    with tf.device('/GPU:0'):
        train_models(train_ds, val_ds, len(class_names), SIZE[0])



    #normalization_layer = layers.Rescaling(1. / 255)
    #train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


