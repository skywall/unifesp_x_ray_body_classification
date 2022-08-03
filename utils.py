import os

import numpy as np
import pydicom
from matplotlib import pyplot as plt


def load_dcm_image(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = img / img.max() * 255

    if dicom.PhotometricInterpretation == "MONOCHROME2":
        img = img.astype("int32")
        img = np.invert(img) + 255

    img = img.astype("float32")

    return img


def get_dcm_filenames_dict(dir):
    filenames = {}

    for (dir_path, dir_names, file_names) in os.walk(dir):
        for file_name in file_names:
            if file_name != '':
                sop_uid = file_name[:-6]  # remove "-c.dcm"
                filenames[sop_uid] = os.path.join(dir_path, file_name)

    return filenames


def plot_images(imgs, labels=None, cols=5, show=True):
    """
    Plot list of images into a grid
    :param imgs: list of images
    :param labels: list of labels
    :param cols: number of columns in plotted grid
    :param show: True if plot should be immediately shown
    :return:
    """
    count = len(imgs)

    if count % cols == 0:
        rows = count // cols
    else:
        rows = (count // cols) + 1

    _, arr = plt.subplots(rows, cols, figsize=(18, 10))
    for idx, image in enumerate(imgs):
        row = idx // cols
        col = idx % cols

        arr[row, col].imshow(imgs[idx])
        arr[row, col].set_xticks([])
        arr[row, col].set_yticks([])
        if labels is not None:
            arr[row, col].set_title(labels[idx])

    plt.tight_layout()
    if show:
        plt.show()


def verify_image_load(show=False):
    # MONOCHROME1
    # img = load_dcm_image(
    #     "dataset_original/train/train/1/1.2.826.0.1.3680043.8.498"
    #     ".89102450329340531816015855773961083133/1.2.826.0.1.3680043.8.498"
    #     ".11278653404499913987623237519434199794/1.2.826.0.1.3680043.8.498"
    #     ".65452424240994805812717428674475343109-c.dcm")

    # MONOCHROME2
    img = load_dcm_image(
        "dataset_original/train/train/936/1.2.826.0.1.3680043.8.498.81043682118573900228148261278330854974/1.2.826.0"
        ".1.3680043.8.498.17655414526918818134087064924097952979/1.2.826.0.1.3680043.8.498"
        ".75354982972213938063778238905438271786-c.dcm")

    if show:
        plt.imshow(img, cmap="gray")
        plt.show()

    assert img.max() <= 255.
    assert img.min() >= 0.
    assert img.shape[0] > 1


def verify_get_dataset_filenames_dict():
    files = get_dcm_filenames_dict("dataset_original/train/train")
    assert len(files.keys()) > 1


if __name__ == '__main__':
    verify_image_load()
    verify_get_dataset_filenames_dict()
