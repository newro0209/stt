import numpy as np


def normalize_audio(chunk: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "int16":
        return chunk.astype(np.float32) / 32768.0
    return chunk.astype(np.float32)


def split_channels(chunk: np.ndarray, channels: int) -> list[np.ndarray]:
    if chunk.ndim == 1 or channels == 1:
        return [chunk.astype(np.float32)]
    return [chunk[:, channel].astype(np.float32) for channel in range(channels)]
