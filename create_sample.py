import pandas as pd
import os

print("Loading large dataset...")
df = pd.read_pickle("data/processed/03_features_engineered.pkl")
print(f"Original size: {len(df)} rows")

# The dashboard only uses max 1000 rows anyway for real-time rendering per participant
# Let's take the first 1000 rows for each participant to make a small sample dataset
sample_df = df.groupby("participant").head(1000).copy()

print(f"Sample size: {len(sample_df)} rows")
sample_df.to_pickle("data/processed/03_features_engineered_sample.pkl")
print("Saved sample dataset.")
