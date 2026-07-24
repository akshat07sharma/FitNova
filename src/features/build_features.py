import os
import numpy as np
import pandas as pd

# Paths setup
INTERIM_PATH = "data/interim"
PROCESSED_PATH = "data/processed"


def build_features():
    print("1. Loading raw pickle files...")
    #piclke files load kar rahe hain
    acc_df = pd.read_pickle(os.path.join(INTERIM_PATH, "01_accelerometer_raw.pkl"))
    gyro_df = pd.read_pickle(os.path.join(INTERIM_PATH, "01_gyroscope_raw.pkl"))

    # 2. Resampling ke liye datetime index banana zaroori hai
    #epochs to datetime mein convert kr rhe hai 
    if "epoch (ms)" in acc_df.columns:
        acc_df["datetime"] = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        
        #datetime ko index bana rahe hain kyuki resampling ke liye zaroori hai
        acc_df = acc_df.set_index("datetime")

    if "epoch (ms)" in gyro_df.columns:
        gyro_df["datetime"] = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
        gyro_df = gyro_df.set_index("datetime")

    print("3. Resampling data to 200ms intervals...")
    # Groupwise resample karke average value le rahe hain
    acc_resampled = (
        acc_df.groupby(["participant", "label", "category"])
        .resample("200ms")
        .mean(numeric_only=True)
        .reset_index()
    )

    gyro_resampled = (
        gyro_df.groupby(["participant", "label", "category"])
        .resample("200ms")
        .mean(numeric_only=True)
        .reset_index()
    )

    print("4. Merging accelerometer and gyroscope datasets...")
    # Time aur labels ke basis par horizontal join
    merged_df = pd.merge(
        acc_resampled,
        gyro_resampled,
        on=["datetime", "participant", "label", "category"],
        suffixes=("_acc", "_gyro"),
    )

    # Missing numeric values fill kar rahe hain
    num_cols = merged_df.select_dtypes(include=["float64", "int64"]).columns
    merged_df[num_cols] = merged_df[num_cols].interpolate(method="linear")

    print("5. Calculating Resultant Magnitude & Rolling Stats...")
    # Accelerometer Magnitude (Total Resultant Force)
    merged_df["acc_r"] = np.sqrt(
        merged_df["x-axis (g)"] ** 2
        + merged_df["y-axis (g)"] ** 2
        + merged_df["z-axis (g)"] ** 2
    )

    # Gyroscope Magnitude (Total Angular Velocity)
    merged_df["gyro_r"] = np.sqrt(
        merged_df["x-axis (deg/s)"] ** 2
        + merged_df["y-axis (deg/s)"] ** 2
        + merged_df["z-axis (deg/s)"] ** 2
    )

    # Rolling Mean & Rolling Std Dev (1 second window = 5 samples)
    merged_df["acc_r_mean"] = (
        merged_df["acc_r"].rolling(window=5, min_periods=1).mean()
    )
    merged_df["acc_r_std"] = (
        merged_df["acc_r"].rolling(window=5, min_periods=1).std().fillna(0)
    )

    # 6. Final file save kar rahe hain
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    output_file = os.path.join(PROCESSED_PATH, "03_features_engineered.pkl")
    merged_df.to_pickle(output_file)

    print("Done! Final dataset saved successfully.")
    return merged_df


if __name__ == "__main__":
    df = build_features()
    print("Final Shape:", df.shape)