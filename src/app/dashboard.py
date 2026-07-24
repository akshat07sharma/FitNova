import sys
import os
import time

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

for path in [SRC_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import joblib
import pandas as pd
import streamlit as st

# Custom module imports from src/utils/
try:
    from utils.calories import calculate_calories
    from utils.rep_counter import count_reps
    from utils.strain import calculate_strain
except ModuleNotFoundError:
    from src.utils.calories import calculate_calories
    from src.utils.rep_counter import count_reps
    from src.utils.strain import calculate_strain

# Page Configuration
st.set_page_config(page_title="FitNova Dashboard", layout="wide")
st.title("🏋️‍♂️ FitNova - AI Fitness Tracker")

# 1. Load Model & Processed Data
@st.cache_resource
def load_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "fitnova_rf_model.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    sample_path = os.path.join(PROJECT_ROOT, "data", "processed", "03_features_engineered_sample.pkl")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "03_features_engineered.pkl")
    if os.path.exists(sample_path):
        df_loaded = pd.read_pickle(sample_path)
    else:
        df_loaded = pd.read_pickle(data_path)
    return df_loaded

try:
    t0 = time.time()
    model = load_model()
    df = load_dataset()
    st.sidebar.caption(f"⚡ Data loaded in {round(time.time() - t0, 2)}s ({len(df):,} total rows)")
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# 2. Sidebar Controls & Performance Safeguard
st.sidebar.header("User Settings")
participant = st.sidebar.selectbox("Select Participant", df["participant"].unique())
user_weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=75.0)

# Limit data size to max 1,000 rows for real-time rendering responsiveness
user_df = df[df["participant"] == participant].copy()
if len(user_df) > 1000:
    st.sidebar.info(f"Subsampled participant dataset from {len(user_df)} to 1,000 rows for smooth UI performance.")
    user_df_sampled = user_df.iloc[:1000]
else:
    user_df_sampled = user_df

# 3. Model Inference (Fast vectorized batch prediction)
feature_cols = [
    "x-axis (g)", "y-axis (g)", "z-axis (g)",
    "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)",
    "acc_r", "gyro_r", "acc_r_mean", "acc_r_std"
]

X_sample = user_df_sampled.reindex(columns=feature_cols, fill_value=0)
predictions = model.predict(X_sample)

# 4. Perform Computations
top_exercise = pd.Series(predictions).mode()[0]
duration_sec = len(user_df_sampled) * 0.2

acc_r_vals = user_df_sampled["acc_r"].values if "acc_r" in user_df_sampled.columns else user_df_sampled.iloc[:, 0].values

# Run computations safely
try:
    total_reps = count_reps(acc_r_vals)
except Exception:
    total_reps = 0

try:
    total_calories = calculate_calories(top_exercise, duration_sec, user_weight)
except Exception:
    total_calories = 0.0

try:
    strain_score = calculate_strain(acc_r_vals, duration_sec)
except Exception:
    strain_score = 0.0

# 5. Render UI Metrics Display
st.subheader(f"Live Tracking for Participant: {participant}")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Exercise", str(top_exercise).upper())
col2.metric("Total Repetitions", f"{total_reps} reps")
col3.metric("Calories Burned", f"{total_calories} kcal")
col4.metric("Session Strain", f"{strain_score} / 21")

# Accelerometer Plot Display (Subsampled for browser responsiveness)
st.subheader("Accelerometer Resultant Magnitude (`acc_r`)")
plot_data = user_df_sampled["acc_r"].iloc[::2] if "acc_r" in user_df_sampled.columns else user_df_sampled.iloc[::2, 0]
st.line_chart(plot_data)