import csv
from collections import defaultdict

import numpy as np
import pandas
import tensorflow
from matplotlib import pyplot as plt

from augmentation import plot_images
from config import BATCH_SIZE, label_as_string
from dataset import XRayDatasetGenerator
from train import create_model


def generate_submission():
    dataset_eval = XRayDatasetGenerator(
        "dataset_generated/test", "dataset_original/sample_submission.csv", training=False, augmentation=False
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

            uids.append(label.numpy().decode())
            targets.append(" ".join(map(str, pred_label)))

    data = {
        "SOPInstanceUID": uids,
        "Target": targets
    }

    df = pandas.DataFrame(data)
    df.to_csv("submission.csv", index=False, quoting=csv.QUOTE_NONE)


def evaluate_on_training_dataset(plot_incorrect=True):
    dataset_train = XRayDatasetGenerator(
        "dataset_generated/train", "dataset_original/train.csv", training=True, augmentation=False
    ).get_dataset().batch(BATCH_SIZE)

    model = create_model()

    latest = tensorflow.train.latest_checkpoint("checkpoints/")
    model.load_weights(latest)

    total = defaultdict(int)
    correct = defaultdict(int)
    incorrect = defaultdict(int)

    incorrect_labels = []
    incorrect_imgs = []

    for (x, y) in dataset_train:
        preds = model(x)
        preds = np.rint(preds)

        for x, y, pred in zip(x, y, preds):
            y_label = np.array(y).nonzero()[0]
            pred_label = pred.nonzero()[0]

            truth_labels_id = ",".join(map(str, y_label))
            pred_labels_id = ",".join(map(str, pred_label))

            y_label_text = np.array([label_as_string(label_id) for label_id in y_label])
            pred_label_text = np.array([label_as_string(label_id) for label_id in pred_label])

            truth_labels_text = ",".join(map(str, y_label_text))
            pred_labels_text = ",".join(map(str, pred_label_text))

            if pred_labels_id == truth_labels_id:
                total[pred_labels_id] = total[pred_labels_id] + 1
                correct[pred_labels_id] = correct[pred_labels_id] + 1
            else:
                total[truth_labels_id] = total[truth_labels_id] + 1
                incorrect[truth_labels_id + " → " + pred_labels_id] = incorrect[truth_labels_id + " → " + pred_labels_id] + 1

                incorrect_labels.append(truth_labels_text + " → " + pred_labels_text)
                incorrect_imgs.append(x / 255)

    total = dict(sorted(total.items()))
    correct = dict(sorted(correct.items()))
    incorrect = dict(sorted(incorrect.items()))

    incorrect_labels = np.array([str(idx) + ": " + label for idx, label in enumerate(incorrect_labels)])

    if plot_incorrect:
        img_cnt = 21
        for idx in range(0, len(incorrect_imgs), img_cnt):
            plot_images(incorrect_imgs[idx:idx+img_cnt], incorrect_labels[idx:idx+img_cnt], cols=7, show=False)

    dicts = [total, correct, incorrect]
    titles = ["Total:", "Correctly classified:", "Incorrectly classified [truth→pred]: "]

    _, arr = plt.subplots(3, 1, figsize=(18, 10))

    for idx, (dct, title) in enumerate(zip(dicts, titles)):
        bars = arr[idx].bar(dct.keys(), dct.values())
        arr[idx].set_title(title + str(sum(dct.values())))
        arr[idx].bar_label(bars)

        arr[idx].tick_params(labelrotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # generate_submission()
    evaluate_on_training_dataset()
