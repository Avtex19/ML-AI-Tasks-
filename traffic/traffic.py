import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")


def load_data(data_dir):
    """
    Load and process image data from directory `data_dir`.

    Returns:
        images: list of resized image arrays
        labels: list of corresponding category labels
    """
    print(f"Reading data from '{data_dir}'...")

    images, labels = [], []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not category.isdigit():
            print(f"Skipping invalid category folder: {category}")
            continue

        for file in os.listdir(category_path):
            filepath = os.path.join(category_path, file)
            img = cv2.imread(filepath)
            if img is not None:
                resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                images.append(resized_img)
                labels.append(int(category))

    if len(images) != len(labels):
        sys.exit("Data mismatch: Number of images doesn't match number of labels.")

    print(f"Successfully loaded {len(images)} labeled images.")
    return images, labels


def get_model():
    """
    Build and compile a CNN model for image classification.

    Returns:
        model: compiled keras model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()
