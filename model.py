import keras
from keras import layers, models

from constants import SIZE
from utils.generate_reports import generate_classification_report


def compile_model(data_dir):
    batch = 32
    epochs = 10
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

    test_ds = keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=SIZE,
        batch_size=batch
    )

    normalization_layer = layers.Rescaling(1. / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    class_names = ['A-1','A-2','A-7','A-11a','A-16','A-17','A-21','A-30','B-1','B-2','B-20','B-21',
                   'B-22','B-23','B-25','B-26','B-33','B-36','B-41','C-2','C-4','C-5','C-9','C-12',
                   'D-1','D-3','D-6','D-6b','D-18','T-27']

    num_classes = len(class_names)

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

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    test_loss, test_acc = model.evaluate(test_ds)
    ballanced_acc = generate_classification_report(model, test_ds, class_names, "sequential", SIZE[0])
    print(f"Dokładność na zbiorze testowym: {test_acc * 100}%")
    return model, ballanced_acc
