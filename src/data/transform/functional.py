import math
from copy import deepcopy
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import butter, lfilter

try:
    import tensorflow as tf
except:
    pass


CV2_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
    "linear_exact": cv2.INTER_LINEAR_EXACT,
    "nearest_exact": cv2.INTER_NEAREST_EXACT,
    "max": cv2.INTER_MAX,
}


def interp_1d(frames, target_len, method="nearest", backend="cv2"):
    """Linear interpolate the temporal dimension:
    T_in * C -> T_out * C
    """
    assert backend in ["cv2", "torch", "tf"]
    assert target_len >= 1
    assert len(frames.shape) == 2
    T, C = frames.shape
    if backend == "tf":
        frames = frames.reshape(-1, 1, C)
        # bilinear, lanczos3, lanczos5, bicubic, gaussian, nearest, area, mitchellcubic
        frames = tf.image.resize(
            frames, (target_len, 1), method=method, antialias=False
        )
        frames = tf.squeeze(frames, axis=1).numpy()
    elif backend == "torch":
        frames = (
            torch.from_numpy(frames).permute(1, 0).unsqueeze(0)
        )  # T*C -> N*C*T (N=1)
        #  'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
        frames = F.interpolate(frames, target_len, mode=method, antialias=False)[
            0
        ].permute(1, 0)
        frames = frames.numpy()
    elif backend == "cv2":
        frames = np.expand_dims(frames, axis=1)  # T*C -> T*1*C
        frames = cv2.resize(
            frames, (1, target_len), interpolation=CV2_INTERPOLATIONS[method]
        ).squeeze(1)
    return frames


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def butter_lowpass_filter(data, fmax=20, sr=200, order=4):
    nyquist = sr / 2
    normal_cutoff = fmax / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data