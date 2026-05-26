import os
import random

from src.utils.config import (
    TRAIN_LIMIT,
    VAL_LIMIT,
    TEST_LIMIT
)

random.seed(42)


def load_audio_paths(base_path, split):

    if split == "train":
        limit = TRAIN_LIMIT

    elif split == "validation":
        limit = VAL_LIMIT

    else:
        limit = TEST_LIMIT

    data = []

    classes = ["real", "fake"]

    for label, class_name in enumerate(classes):

        folder = os.path.join(
            base_path,
            split,
            class_name
        )

        files = os.listdir(folder)

        random.shuffle(files)

        if limit is not None:
            files = files[:limit]

        for file in files:

            full_path = os.path.join(
                folder,
                file
            )

            data.append(
                (full_path, label)
            )

    random.shuffle(data)

    return data