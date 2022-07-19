import csv
from collections import defaultdict

import numpy as np
import pandas
import tensorflow
from matplotlib import pyplot as plt

from config import BATCH_SIZE
from dataset import XRayDatasetGenerator
from train import create_model


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

            uids.append(label.numpy().decode())
            targets.append(" ".join(map(str, pred_label)))

    data = {
        "SOPInstanceUID": uids,
        "Target": targets
    }

    df = pandas.DataFrame(data)
    df.to_csv("submission.csv", index=False, quoting=csv.QUOTE_NONE)


def evaluate_on_training_dataset():
    dataset_train = XRayDatasetGenerator(
        "dataset_generated/train", "dataset_original/train.csv", training=True
    ).get_dataset().batch(BATCH_SIZE)

    model = create_model()

    latest = tensorflow.train.latest_checkpoint("checkpoints/")
    model.load_weights(latest)

    total = defaultdict(int)
    correct = defaultdict(int)
    incorrect = defaultdict(int)

    for (x, y) in dataset_train:
        preds = model(x)
        preds = np.rint(preds)

        for y, pred in zip(y, preds):
            y_label = np.array(y).nonzero()[0]
            pred_label = pred.nonzero()[0]

            pred_labels = ",".join(map(str, pred_label))
            truth_labels = ",".join(map(str, y_label))

            if pred_labels == truth_labels:
                total[pred_labels] = total[pred_labels] + 1
                correct[pred_labels] = correct[pred_labels] + 1
            else:
                total[truth_labels] = total[truth_labels] + 1
                incorrect[truth_labels + "->" + pred_labels] = incorrect[truth_labels + "->" + pred_labels] + 1

    total = dict(sorted(total.items()))
    correct = dict(sorted(correct.items()))
    incorrect = dict(sorted(incorrect.items()))

    _, arr = plt.subplots(3, 1)
    bars = arr[0].bar(total.keys(), total.values())
    arr[0].set_title("Total (" + str(sum(total.values())) + ")")
    arr[0].bar_label(bars)
    bars = arr[1].bar(correct.keys(), correct.values())
    arr[1].set_title("Correctly classified (" + str(sum(correct.values())) + ")")
    arr[1].bar_label(bars)
    bars = arr[2].bar(incorrect.keys(), incorrect.values())
    arr[2].set_title("Incorrectly classified [truth->prediction] (" + str(sum(incorrect.values())) + ")")
    arr[2].bar_label(bars)
    arr[2].tick_params(labelrotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # generate_submission()
    evaluate_on_training_dataset()
