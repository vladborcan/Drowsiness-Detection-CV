"""Constante comune proiectului."""

# UTA-RLDD: eticheta se afla in numele fisierului video (0.mp4 / 5.mov / 10.mp4)
LABEL_TO_INDEX = {0: 0, 5: 1, 10: 2}
CLASS_NAMES = ["alert", "low_vigilance", "drowsy"]

DEFAULT_SEED = 42
SPLITS = ("train", "val", "test")