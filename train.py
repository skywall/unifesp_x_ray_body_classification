import os
from datetime import datetime

import keras
import numpy as np
import tensorflow
from keras import layers, optimizers, losses, callbacks
from keras.applications.resnet_rs import ResNetRS50

from config import LABEL_COUNT, LEARNING_RATE, EPOCHS, BATCH_SIZE, INPUT_IMAGE_SIZE
from dataset import XRayDatasetGenerator
from f1_score import F1Score

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)


def create_model():
    backbone = ResNetRS50(include_top=False, input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))

    model = keras.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(LABEL_COUNT, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy", F1Score()],
    )

    return model


def train(load_weights=False):
    generator_train = XRayDatasetGenerator("dataset_generated/train", "dataset_original/train.csv")

    dataset_train = generator_train.get_dataset()
    dataset_val = dataset_train.take(340).batch(BATCH_SIZE)
    dataset_train = dataset_train.skip(340).shuffle(BATCH_SIZE * 10, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    model = create_model()

    if load_weights:
        latest = tensorflow.train.latest_checkpoint("checkpoints/")
        model.load_weights(latest)

    cp_callback = callbacks.ModelCheckpoint(
        filepath="checkpoints/checkpoint.cp",
        save_weights_only=True,
        verbose=1)

    log_dir = "logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)

    model.fit(
        x=dataset_train,
        epochs=EPOCHS,
        callbacks=[cp_callback, tb_callback, es_callback],
        workers=3,
        validation_data=dataset_val
    )


if __name__ == "__main__":
    train()
