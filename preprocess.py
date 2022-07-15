import math
import os

import cv2 as cv
import pandas
from tqdm import tqdm

from config import LABEL_COUNT, labels_reversed
from utils import get_dcm_filenames_dict, load_dcm_image


def preprocess_train_df(df: pandas.DataFrame):
    """
    Replace Target column by category columns for all
    available classes [Abdomen, Ankle, CervicalSpine, ...]

    :param df: modified dataframe
    :return: preprocessed dataframe
    """

    for class_id in range(LABEL_COUNT + 1):
        df[class_id] = 0

    def fill_classes(row):
        targets = row['Target'].split(" ")
        for target in targets:
            if target != "":
                row[int(target)] = 1

        return row

    df = df.apply(fill_classes, axis=1)
    df = df.drop(columns=["Target"])

    # Rename class_ids to class names: 1 -> Ankle etc.
    df.rename(columns=labels_reversed, inplace=True)
    return df


def convert_dcm_dataset_to_jpg(src_dir, dest_dir, max_image_size=800):
    """
    Convert DICOM files into JPGs. Algorithm walks through src_dir hierarchy and process all the files. Directory
    structure is not preserved, final images are stored directly in dest_dir. Image ratio is preserved.

    :param src_dir: source directory.
    :param dest_dir: destination directory
    :param max_image_size: maximal exported image size
    """
    dcm_uid_to_path = get_dcm_filenames_dict(src_dir)

    os.makedirs(dest_dir, exist_ok=True)

    with tqdm(total=len(dcm_uid_to_path.items())) as pbar:
        for (dcm_uid, filepath) in dcm_uid_to_path.items():
            if os.path.isfile(os.path.join(dest_dir, dcm_uid + ".jpg")):
                pbar.update(1)
                continue

            img = load_dcm_image(filepath)
            h, w = img.shape
            max_image_size = max_image_size

            if h > w:
                ratio = w / h
                h = max_image_size
                w = math.floor(h * ratio)
            else:
                ratio = h / w
                w = max_image_size
                h = math.floor(w * ratio)

            resized = cv.resize(img, dsize=(w, h), interpolation=cv.INTER_AREA)
            cv.imwrite(os.path.join(dest_dir, dcm_uid + ".jpg"), resized)
            pbar.update(1)


if __name__ == '__main__':
    # convert_dcm_dataset_to_jpg("dataset_original/train/", "dataset_generated/train")
    convert_dcm_dataset_to_jpg("dataset_original/test/", "dataset_generated/test")
