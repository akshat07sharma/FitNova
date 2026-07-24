import pandas as pd

# MET values for different exercise intensities
MET_VALUES = {
    "bench": 6.0,
    "squat": 8.0,
    "row": 7.0,
    "ohp": 6.0,
    "dead": 8.0,
    "rest": 1.3,
    "unknown": 3.0
}

def calculate_calories(exercise_name: str, duration_sec: float, weight_kg: float = 75.0) -> float:
    """
    Calculates calories burned based on exercise MET value, duration, and user weight.
    Formula: Calories = MET * Weight (kg) * Duration (hours)
    """
    exercise_key = str(exercise_name).lower()
    
    # Match MET value based on substring
    met = 3.0  # default fallback
    for key, val in MET_VALUES.items():
        if key in exercise_key:
            met = val
            break
            
    duration_hours = duration_sec / 3600.0
    calories = met * weight_kg * duration_hours
    return round(calories, 2)