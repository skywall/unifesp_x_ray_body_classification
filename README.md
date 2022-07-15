# UNIFESP X-ray Body Part Classifier Competition

Kaggle competition [link](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier).

## Data preprocessing

Data, X-Ray images, are provided in `*.dcm` (DICOM) format. Every file contains metadata and pixel array. Since single
file load takes plenty of time (~2s) it is handy to convert them into `*.jpg`.
Function `preprocess.convert_dcm_dataset_to_jpg` could be used for dataset conversion.

It is expected such datasets exist in directories `dataset_generated/train` & `dataset_generated/test` before training.
Images in these directories should follow `<SOPInstanceUID>.jpg` naming convention.

## Experiments

## Baseline

- Train/Validation dataset split: 1588/150
- Model: ResnetRS50 `(224, 224, 3)`
- No shuffle
- F1 Train accuracy: `0.8860`
- F1 Validation accuracy: `0.8133`
- Kaggle test score: `0.78058`

## Experiment #1: Overtrain model

- Epochs: 5 --> 10
- F1 Train accuracy: `0.953`
- F1 Validation accuracy: `0.8533`
- Kaggle test score: `0.79405`

## Experiment #2: ...