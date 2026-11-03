import numpy as np


def window(tot_len: int, max_len: int, stride: int) -> np.ndarray:
    l = np.arange(0, tot_len, stride) if stride > 0 else np.array([0])
    r = np.clip(l + max_len, 0, tot_len)
    return np.stack([l, r], axis=1)
