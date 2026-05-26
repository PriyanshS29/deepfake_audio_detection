from src.utils.data_loader import load_audio_paths

data = load_audio_paths(
    "data/raw",
    "train"
)

print(len(data))

print(data[:5])

