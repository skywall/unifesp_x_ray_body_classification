import numpy as np
import tensorflow as tf
from keras.metrics import Recall, Precision
from sklearn.metrics import f1_score


# F1Score metric from tfa is not able to handle multi-label input
# https://github.com/tensorflow/addons/issues/746
# Code taken from: https://stackoverflow.com/a/64477522/1103478
class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision(thresholds=threshold)
        self.recall_fn = Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 * ((p * r) / (p + r)))

    def result(self):
        return self.f1

    def reset_state(self):
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


def check_f1_score_computation():
    pred = np.array([0.8, 0.1, 0.7, 0.51])
    truth = np.array([1, 0, 1, 0])

    fn = F1Score()
    fn.update_state(truth, pred)

    assert f1_score(truth, np.rint(pred)) - float(fn.result()) < 1e-4


if __name__ == "__main__":
    check_f1_score_computation()
