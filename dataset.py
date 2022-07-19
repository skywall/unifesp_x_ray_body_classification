import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from PIL import Image

from config import INPUT_IMAGE_SIZE, LABEL_COUNT
from preprocess import preprocess_train_df

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class XRayDatasetGenerator:
    def __init__(self, path_to_images, path_to_csv, training=True):
        self.path_to_csv = path_to_csv
        self.path_to_images = path_to_images
        self.training = training

        self.df = pd.read_csv(path_to_csv)
        self.df = preprocess_train_df(self.df)

    def _generator(self):
        for _, row in self.df.iterrows():
            label = str(row["SOPInstanceUID"])

            # read image
            image_path = os.path.join(self.path_to_images, label + ".jpg")
            img = np.asarray(Image.open(image_path))
            img = (img / img.max()) * 255
            img = cv.resize(img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
            img = cv.merge((img, img, img))
            x = img.astype("float32")

            # read targets - skip uid & path
            y = np.array(row[1:], dtype="float32")

            if self.training:
                yield x, y
            else:
                yield x, label

    def get_dataset(self):
        """
        Build dataset.

        :return: tf.data.Dataset filled by generator. If training=True dataset provides image data (x) and labels (y).
        If training=False image data (x) and uid (image identifier) is provided.
        """

        if self.training:
            second_tensor_spec = tensorflow.TensorSpec(shape=(LABEL_COUNT,), dtype=tensorflow.float32)
        else:
            second_tensor_spec = tensorflow.TensorSpec(shape=(), dtype=tensorflow.string)

        return tensorflow.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tensorflow.TensorSpec(shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=tensorflow.float32),
                second_tensor_spec,
            )
        )


def check_training_dataset_load():
    loader = XRayDatasetGenerator("dataset_generated/train", "dataset_original/train.csv")
    dataset = loader.get_dataset()

    for (x, y) in dataset.skip(2).take(1):
        print(x)
        x = x / 255.
        plt.imshow(x, cmap="gray")
        plt.show()


def check_eval_dataset_load():
    loader = XRayDatasetGenerator("dataset_generated/test", "dataset_original/sample_submission.csv", training=False)
    dataset = loader.get_dataset()

    for (x, label) in dataset.take(10):
        print(x, label)


if __name__ == '__main__':
    check_training_dataset_load()
    # check_eval_dataset_load()
