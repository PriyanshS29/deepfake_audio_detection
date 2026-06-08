import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 20
MAX_LEN = 173

def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode='constant'
        )
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = mfcc.T

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    return mfcc