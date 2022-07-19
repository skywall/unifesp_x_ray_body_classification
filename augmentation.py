import albumentations as A
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

transform = A.Compose([
    A.Rotate(border_mode=cv.BORDER_CONSTANT, value=(255, 255, 255), limit=15),
    A.CLAHE(),
    A.OneOf([
        A.GaussNoise(),
        A.ISONoise(),
    ]),
    A.RandomBrightnessContrast(),
    A.VerticalFlip()
])


def plot_images(imgs, labels=None, cols=5):
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
        if labels is not None:
            arr[row, col].set_title(labels[idx])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = cv.imread("dataset_generated/train/1.2.826.0.1.3680043.8.498.10036150326276641158002573300029848125.jpg")
    img = np.array(img)

    IMG_COUNT = 15
    imgs = [transform(image=img)["image"] for i in range(IMG_COUNT)]

    plot_images(imgs)
