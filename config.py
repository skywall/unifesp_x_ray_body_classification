LEARNING_RATE = 0.0001
EPOCHS = 12
INPUT_IMAGE_SIZE = 320
BATCH_SIZE = 32

LABEL_COUNT = 22

labels = {
    "Abdomen": 0,
    "Ankle": 1,
    "CervicalSpine": 2,
    "Chest": 3,
    "Clavicles": 4,
    "Elbow": 5,
    "Feet": 6,
    "Finger": 7,
    "Forearm": 8,
    "Hand": 9,
    "Hip": 10,
    "Knee": 11,
    "LowerLeg": 12,
    "LumbarSpine": 13,
    "Others": 14,
    "Pelvis": 15,
    "Shoulder": 16,
    "Sinus": 17,
    "Skull": 18,
    "Thigh": 19,
    "ThoracicSpine": 20,
    "Wrist": 21,
}

labels_reversed = dict((v, k) for k, v in labels.items())
