import os
import librosa
import numpy as np

from tqdm import tqdm

from src.utils.data_loader import load_audio_paths

from src.utils.config import (
    SAMPLE_RATE,
    MAX_PAD_LEN
)

BASE_PATH = "data/raw"

SAVE_PATH = "data/features/mel_spectrogram"


def extract_mel_spectrogram(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    mel_spec = librosa.feature.melspectrogram(

        y=audio,
        sr=sample_rate,
        n_mels=128

    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # Padding

    if mel_spec_db.shape[1] < MAX_PAD_LEN:

        pad_width = (
            MAX_PAD_LEN -
            mel_spec_db.shape[1]
        )

        mel_spec_db = np.pad(

            mel_spec_db,

            pad_width=((0, 0), (0, pad_width)),

            mode='constant'

        )

    # Trimming

    else:

        mel_spec_db = mel_spec_db[:, :MAX_PAD_LEN]

    return mel_spec_db


for split in ["train", "validation", "test"]:

    print(f"\nProcessing {split} data...\n")

    data = load_audio_paths(
        BASE_PATH,
        split
    )

    for file_path, label in tqdm(data):

        class_name = (
            "real"
            if label == 0
            else "fake"
        )

        save_folder = os.path.join(

            SAVE_PATH,
            split,
            class_name

        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        feature = extract_mel_spectrogram(
            file_path
        )

        filename = os.path.basename(
            file_path
        )

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

print(
    "\nMel Spectrogram Extraction Completed"
)