import glob
import os
import pandas as pd

# Path set kar rahe hain jahan raw data aur interim files save hongi
DATA_PATH = "data/MetaMotion"
INTERIM_PATH = "data/interim"


def load_raw_data():
    # MetaMotion folder se saare CSV files ke paths collect kar rahe hain
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

    print(f"Total CSV Files Found: {len(all_files)}")

    # Dono sensors ke data ko alag hold karne ke liye empty lists banayi hain
    acc_list = []
    gyro_list = []

    # Har ek CSV file par loop chala kar data process karenge
    for file in all_files:
        # File name nikal rahe hain bina extension ke (e.g. B-ohp-heavy2-rpe7_...)
        filename = os.path.basename(file).replace(".csv", "")

        # Target metadata extract karne ke liye string split kar rahe hain
        parts = filename.split("_")
        meta = parts[0].split("-")

        # Hyphen se split karke initial parameters nikal rahe hain
        participant = meta[0] if len(meta) > 0 else "unknown"
        label = meta[1] if len(meta) > 1 else "unknown"
        category = meta[2] if len(meta) > 2 else "unknown"

        # CSV file read kar rahe hain pandas dataframe me
        df = pd.read_csv(file)

        # Dataset ke andar extra columns attach kar rahe hain identification ke liye
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Filename check karke decide kar rahe hain ki ye Accelerometer hai ya Gyroscope
        if "Accelerometer" in filename:
            acc_list.append(df)
        elif "Gyroscope" in filename:
            gyro_list.append(df)

    # Saari choti CSVs ke dataframes ko ek bade dataframe me jod (combine) rahe hain
    acc_df = pd.concat(acc_list, ignore_index=True)
    gyro_df = pd.concat(gyro_list, ignore_index=True)

    # Folder banayenge agar pehle se exist nahi karta toh
    os.makedirs(INTERIM_PATH, exist_ok=True)

    # Processed tables ko save kar rahe hain pickle format me
    acc_df.to_pickle(os.path.join(INTERIM_PATH, "01_accelerometer_raw.pkl"))
    gyro_df.to_pickle(os.path.join(INTERIM_PATH, "01_gyroscope_raw.pkl"))

    print("Data loading done! Accelerometer and Gyroscope files saved successfully.")

    return acc_df, gyro_df


if __name__ == "__main__":
    acc_df, gyro_df = load_raw_data()
    print("Accelerometer Data Shape:", acc_df.shape)
    print("Gyroscope Data Shape:", gyro_df.shape)