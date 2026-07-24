import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"


def train_fitnova_model():
    print("1. Loading feature-engineered dataset...")
    df = pd.read_pickle(os.path.join(PROCESSED_PATH, "03_features_engineered.pkl"))

    print("2. Selecting Features (X) and Target Label (y)...")
    feature_cols = [
        "x-axis (g)",
        "y-axis (g)",
        "z-axis (g)",
        "x-axis (deg/s)",
        "y-axis (deg/s)",
        "z-axis (deg/s)",
        "acc_r",
        "gyro_r",
        "acc_r_mean",
        "acc_r_std",
    ]

    X = df[feature_cols].fillna(0).copy()
    y = df["label"]

    # Sensor noise simulation
    np.random.seed(42)
    noise = np.random.normal(0, 0.45, X.shape)
    X = X + noise

    print("3. Splitting Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("4. Training Balanced Constrained Random Forest...")
    # class_weight='balanced' use kar rahe hain taaki minor classes (dead, ohp, squat) miss na ho
    model = RandomForestClassifier(
        n_estimators=15,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("5. Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n---> Model Accuracy: {accuracy * 100:.2f}%\n")
    print(
        "Classification Report:\n",
        classification_report(y_test, y_pred, zero_division=0),
    )

    print("6. Saving trained model to disk...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    saved_file = os.path.join(MODEL_PATH, "fitnova_rf_model.pkl")
    joblib.dump(model, saved_file)

    print(f"Model saved successfully at: {saved_file}")


if __name__ == "__main__":
    train_fitnova_model()