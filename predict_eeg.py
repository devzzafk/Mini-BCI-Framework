import numpy as np
import pandas as pd
import joblib

# --- Step 1: Simulate a new EEG session ---
fs = 250  # samples per second
seconds = 5
channels = 4
time = np.arange(0, seconds, 1/fs)

# Random new EEG signal
new_data = np.random.randn(len(time), channels)
df_new = pd.DataFrame(new_data, columns=[f'Ch{i+1}' for i in range(channels)])

# --- Step 2: Extract features (mean & variance) ---
features = {}
for col in df_new.columns:
    features[f"{col}_mean"] = df_new[col].mean()
    features[f"{col}_var"] = df_new[col].var()

features_df = pd.DataFrame([features])

# --- Step 3: Load trained model ---
model = joblib.load("../models/eeg_model.pkl")

# --- Step 4: Predict ---
prediction = model.predict(features_df)[0]

# Convert 0/1 to text
label = "Relaxed" if prediction == 0 else "Focused"

print("Prediction for new EEG session:", label)
