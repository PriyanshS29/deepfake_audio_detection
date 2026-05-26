import os
import librosa
import numpy as np

from tqdm import tqdm

from src.utils.data_loader import load_audio_paths

from src.utils.config import (
    SAMPLE_RATE,
    N_MFCC,
    MAX_PAD_LEN
)

BASE_PATH = "data/raw"

SAVE_PATH = "data/features/mfcc"


def extract_mfcc(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Padding if smaller

    if mfcc.shape[1] < MAX_PAD_LEN:

        pad_width = MAX_PAD_LEN - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant'
        )

    # Trimming if larger

    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc


for split in ["train", "validation", "test"]:

    print(f"\nProcessing {split} data...\n")

    data = load_audio_paths(
        BASE_PATH,
        split
    )

    for file_path, label in tqdm(data):

        class_name = "real" if label == 0 else "fake"

        save_folder = os.path.join(
            SAVE_PATH,
            split,
            class_name
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        feature = extract_mfcc(file_path)

        filename = os.path.basename(file_path)

        filename = filename.replace(
            ".wav",
            ".npy"
        )

        save_file = os.path.join(
            save_folder,
            filename
        )

        np.save(
            save_file,
            feature
        )

print("\nMFCC Extraction Completed Successfully")