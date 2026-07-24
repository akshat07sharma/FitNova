import numpy as np
from scipy.signal import find_peaks

def count_reps(acc_r_series: np.ndarray, height_threshold: float = None, distance: int = 5) -> int:
    """
    Counts repetitions using peak detection on resultant acceleration (acc_r).
    200ms sample frequency -> distance parameter prevents counting noise within ~1 sec.
    """
    if len(acc_r_series) == 0:
        return 0

    # Auto-threshold if not provided (mean + standard deviation)
    if height_threshold is None:
        height_threshold = np.mean(acc_r_series) + np.std(acc_r_series) * 0.5

    peaks, _ = find_peaks(acc_r_series, height=height_threshold, distance=distance)
    return int(len(peaks))