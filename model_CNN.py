import os
import keras
from keras import layers, models
from Utils.constants import SIZE, BATCH_SIZE, SIGN_NAME, EPOCHS, GRAPH, MODELS_DICT_SEQUENTIAL
from Utils.generate_reports import generate_classification_report
from Utils.graphs import graph_creator


def compile_model(data_dir):
    data = data_dir

    train_ds = keras.utils.image_dataset_from_directory(
        f"{data}/train",
        image_size=SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=SIZE,
        batch_size=BATCH_SIZE,
    )

    normalization_layer = MODELS_DICT_SEQUENTIAL["sequential"]
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    num_classes = len(SIGN_NAME)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE[0], SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if GRAPH:
        graph_creator(history, "sequential")

    path_to_save = os.path.join("models", "sequential")
    os.makedirs("models", exist_ok=True)
    os.makedirs(path_to_save, exist_ok=True)

    bal_acc = generate_classification_report(model, val_ds, SIGN_NAME, "sequential", SIZE[0], path_to_save)
    model_path = os.path.join(path_to_save, f"{bal_acc:.3f}.keras")
    model.save(model_path)
