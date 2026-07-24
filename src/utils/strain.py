import numpy as np

def calculate_strain(acc_r_series: np.ndarray, duration_sec: float) -> float:
    """
    Calculates a workload strain score (0.0 to 21.0 scale, WHOOP-style)
    based on average acceleration force and total session volume/duration.
    """
    if len(acc_r_series) == 0 or duration_sec == 0:
        return 0.0

    avg_force = np.mean(acc_r_series)
    std_force = np.std(acc_r_series)
    
    # Composite strain logic based on intensity and time
    raw_score = (avg_force * 0.4 + std_force * 0.6) * (duration_sec / 60.0)
    
    # Normalize score on a 0 - 21 scale
    strain_score = min(21.0, max(0.0, raw_score / 5.0))
    return round(strain_score, 1)