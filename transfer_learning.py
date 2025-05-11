import os
import keras
import tensorflow as tf
from Utils.constants import SIZE, MODELS_DICT_TRANSFER, EPOCHS, BATCH_SIZE, SIGN_NAME, GRAPH
from Utils.generate_reports import generate_classification_report
from Utils.graphs import graph_creator
from keras import layers, models
from keras.src.layers import GlobalAveragePooling2D
from keras.src.optimizers import Adam


def train_models(train_data, val_data, num_classes, img_size, epochs=EPOCHS, batch_size=BATCH_SIZE):
    preprocessed_datasets = {}
    for model_name, (_, preprocess_fn) in MODELS_DICT_TRANSFER.items():
        preprocessed_datasets[model_name] = (
            train_data.map(lambda x, y: (preprocess_fn(x), y)),
            val_data.map(lambda x, y: (preprocess_fn(x), y))
        )

    for model_name, (fn_model, _) in MODELS_DICT_TRANSFER.items():
        print(f"\nTraining model: {model_name} for img sizes: {img_size}x{img_size}")

        train_data_processed, val_data_processed = preprocessed_datasets[model_name]

        base_model = fn_model(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_data_processed,
                            validation_data=val_data_processed,
                            epochs=epochs,
                            batch_size=batch_size)


        path_to_save = os.path.join("models", model_name)
        os.makedirs("models", exist_ok=True)
        os.makedirs(path_to_save, exist_ok=True)

        bal_acc = generate_classification_report(model, val_data_processed, SIGN_NAME,
                                                  model_name, SIZE[0], path_to_save)

        if GRAPH:
            graph_creator(history, model_name)

        model_path = os.path.join(path_to_save, f"{bal_acc:.3f}.keras")
        model.save(model_path)

def compile_model_transfer_learning(data_dir):
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
        train_models(train_ds, val_ds, len(SIGN_NAME), SIZE[0])

