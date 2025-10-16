import numpy as np
import cv2
from typing import List, Tuple
from scipy.ndimage import gaussian_filter


def compute_channels(image: np.ndarray) -> np.ndarray:
    """
    image: [H, W, 3] (RGB)
    out: [H, W, 10]
    """
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    L, U, V = cv2.split(luv)

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    num_bins = 6
    orientation = np.mod(orientation, np.pi)
    bin_width = np.pi / num_bins
    hog_channels = []

    for i in range(num_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width
        mask = (orientation >= lower) & (orientation < upper)
        hog_channels.append(magnitude * mask.astype(np.float32))

    channels = [L, U, V, magnitude] + hog_channels
    return np.stack(channels, axis=-1)


def smooth_channels(channels, sigma=1):
    smoothed = np.zeros_like(channels)
    for i in range(channels.shape[-1]):
        smoothed[..., i] = gaussian_filter(channels[..., i], sigma=sigma)
    return smoothed


def aggregate_channels(channels: np.ndarray, feature_resolution: int) -> np.ndarray:
    """
    channels: [H, W, C]
    out: [feature_resolution, feature_resolution, C]
    """
    _, _, C = channels.shape
    aggregated = np.zeros((feature_resolution, feature_resolution, C), dtype=np.float32)

    for i in range(C):
        aggregated[..., i] = cv2.resize(
            channels[..., i],
            (feature_resolution, feature_resolution),
            interpolation=cv2.INTER_AREA,
        )

    return aggregated


def compute_channel_pyramid(
    image: np.ndarray, scales: List[float]
) -> List[Tuple[np.ndarray, float]]:
    """
    image: [H, W, C] (RGB)
    out: List of [H * scale, W * scale, 3], scale
    """
    pyramid = []

    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        if new_h < 32 or new_w < 32:  # skip very small scales
            continue

        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pyramid.append((scaled_img, scale))

    return pyramid
