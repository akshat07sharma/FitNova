# 🏋️ FitNova

**FitNova** is a workout-tracking and analytics pipeline built on MetaMotion sensor data. It loads raw accelerometer/gyroscope readings, engineers features, detects the exercise being performed, counts repetitions, estimates calories burned, computes a training strain score, and presents everything in an interactive dashboard.

🔗 **Live Demo:** [fitnova-07.streamlit.app](https://fitnova-07.streamlit.app/)

---

## ✨ Features

- 📥 **Automated data ingestion** from raw MetaMotion CSV sensor files
- ⚙️ **Feature engineering** on accelerometer/gyroscope signals (magnitudes, rolling stats)
- 🤖 **Exercise recognition** via a trained `RandomForestClassifier`
- 🔢 **Rep counting** using peak detection on smoothed sensor signals
- 🔥 **Calorie estimation** using MET-based calculations
- 💪 **Training strain score** (TRIMP-style load metric)
- 📊 **Interactive Streamlit dashboard** to visualize it all in one place

---

## 🚀 Live Demo

Try FitNova without installing anything:

👉 **[https://fitnova-07.streamlit.app/](https://fitnova-07.streamlit.app/)**

---

## 📁 Project Structure

```
FitNova/
│
├── data/
│   └── MetaMotion/
│       ├── A-bench-heavy1.csv
│       ├── A-bench-heavy2.csv
│       ├── A-dead-heavy1.csv
│       ├── A-row-medium1.csv
│       ├── ...
│
├── models/
│   └── activity_model.pkl          # Trained activity-recognition model
│
├── src/
│   ├── data/
│   │   └── load_data.py            # Load & combine raw MetaMotion CSVs
│   │
│   ├── features/
│   │   └── build_features.py       # Engineer model-ready features
│   │
│   ├── models/
│   │   ├── train_model.py          # Train the activity classifier
│   │   └── predict_model.py        # Run predictions on new data
│   │
│   ├── utils/
│   │   ├── rep_counter.py          # Count reps from sensor signals
│   │   ├── calories.py             # Estimate calories burned (MET-based)
│   │   └── strain.py               # Compute workout strain score
│   │
│   └── app/
│       └── dashboard.py            # Streamlit analytics dashboard
│
├── requirements.txt
├── README.md
└── main.py                         # CLI entry point
```

---

## 🛠️ Setup

1. **Clone or unzip the project.**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your sensor data.** Place raw MetaMotion sensor CSV files in `data/MetaMotion/`.
   Filenames are expected in the form:
   `<participant>-<exercise>-<category><set_number>.csv`
   (e.g. `A-bench-heavy1.csv`).

---

## ▶️ Usage

Run these commands from the project root.

**1. Load & inspect raw sensor data:**
```bash
python main.py load
```

**2. Build engineered features** (saves `data/metamotion_features.pkl`):
```bash
python main.py features
```

**3. Train the activity recognition model:**
```bash
python main.py train
```
Reads `data/metamotion_features.pkl`, trains a classifier, and saves it to `models/activity_model.pkl`.

**4. Run predictions:**
```bash
python main.py predict
```

**5. Launch the dashboard locally:**
```bash
streamlit run src/app/dashboard.py
```

Or just use the **[hosted version](https://fitnova-07.streamlit.app/)** — no setup required.

---

## 📦 Modules

| Module | Description |
|---|---|
| `data/load_data.py` | Reads all CSVs in `data/MetaMotion/`, parses participant/exercise/category/set from each filename, and concatenates everything into one tidy DataFrame. |
| `features/build_features.py` | Computes resultant accelerometer/gyroscope magnitudes and rolling mean/std features per set, then saves the result to `data/metamotion_features.pkl`. |
| `models/train_model.py` | Trains a `RandomForestClassifier` on the engineered features, evaluates accuracy, and saves the model. |
| `models/predict_model.py` | Loads the saved model and runs inference on new feature rows. |
| `utils/rep_counter.py` | Applies a low-pass filter to smooth sensor signals and detects peaks to count repetitions per set. |
| `utils/calories.py` | Estimates calories burned using MET (Metabolic Equivalent of Task) values combined with user weight and duration. |
| `utils/strain.py` | Computes a TRIMP-style training strain/load score based on heart rate and duration. |
| `app/dashboard.py` | A Streamlit app that ties everything together: upload sensor data, view detected reps, calories, and strain score. |

---

## 📝 Notes

- This is a starter template with sample MetaMotion CSVs included for reference — replace them with your real sensor data.
- Customize `MET_VALUES` in `utils/calories.py`, the model choice in `models/train_model.py`, and the strain formula in `utils/strain.py` to match your dataset and goals.
- Sampling rate assumptions (e.g., ~20Hz) in `utils/rep_counter.py` and `app/dashboard.py` should be adjusted to match your actual sensor's sampling frequency.
- The filename parser in `data/load_data.py` assumes the pattern `<participant>-<exercise>-<category><set>.csv`; adjust the regex if your naming convention differs.

---

## 🔗 Links

- **Live App:** [https://fitnova-07.streamlit.app/](https://fitnova-07.streamlit.app/)
