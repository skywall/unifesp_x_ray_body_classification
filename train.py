import csv
import os
from datetime import datetime

import keras
import numpy as np
import pandas
import tensorflow
from keras import layers, optimizers, losses
from keras.applications.resnet_rs import ResNetRS50

from config import LABEL_COUNT, LEARNING_RATE, EPOCHS, BATCH_SIZE
from dataset import XRayDatasetGenerator
from f1_score import F1Score

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)


def create_model():
    backbone = ResNetRS50(include_top=False, input_shape=(224, 224, 3))

    model = keras.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
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
    dataset_val = dataset_train.take(150).batch(BATCH_SIZE)
    dataset_train = dataset_train.skip(150).batch(BATCH_SIZE)

    model = create_model()

    if load_weights:
        latest = tensorflow.train.latest_checkpoint("checkpoints/")
        model.load_weights(latest)

    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/checkpoint.cp",
        save_weights_only=True,
        verbose=1)

    log_dir = "logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

    es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

    model.fit(
        x=dataset_train,
        epochs=EPOCHS,
        callbacks=[cp_callback, tb_callback, es_callback],
        workers=3,
        validation_data=dataset_val
    )


def generate_submission():
    dataset_eval = XRayDatasetGenerator(
        "dataset_generated/test", "dataset_original/sample_submission.csv", training=False
    ).get_dataset().batch(BATCH_SIZE)

    model = create_model()

    latest = tensorflow.train.latest_checkpoint("checkpoints/")
    model.load_weights(latest)

    uids = []
    targets = []

    for (x, labels) in dataset_eval:
        preds = model(x)
        preds = np.rint(preds)

        for label, pred in zip(labels, preds):
            pred_label = pred.nonzero()[0]
            pred_label = pred_label + 1  # labels indexing starts from 1

            uids.append(label.numpy().decode())
            targets.append(" ".join(map(str, pred_label)))

    data = {
        "SOPInstanceUID": uids,
        "Target": targets
    }

    df = pandas.DataFrame(data)
    df.to_csv("submission.csv", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    generate_submission()
