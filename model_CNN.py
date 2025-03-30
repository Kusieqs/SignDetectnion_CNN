import keras
from keras import layers, models
from Utils.constants import SIZE, BATCH_SIZE, CLASS_NAMES, EPOCHS
from Utils.generate_reports import generate_classification_report


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
    normalization_layer = layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    num_classes = len(CLASS_NAMES)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE[0], SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(val_ds)
    ballanced_acc = generate_classification_report(model, val_ds, CLASS_NAMES, "sequential", SIZE[0], 'models')
    print(f"Dokładność na zbiorze testowym: {test_acc * 100}%")
    return model, ballanced_acc