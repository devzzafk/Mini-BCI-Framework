import numpy as np
import pandas as pd
import joblib
import time

# Load trained model
model = joblib.load("../models/eeg_model.pkl")

fs = 250  # samples per second
seconds = 5
channels = 4

print("Starting real-time EEG predictions (Ctrl+C to stop)...\n")

while True:
    # --- Simulate new EEG session ---
    time_points = np.arange(0, seconds, 1/fs)
    new_data = np.random.randn(len(time_points), channels)
    df_new = pd.DataFrame(new_data, columns=[f'Ch{i+1}' for i in range(channels)])

    # --- Extract features ---
    features = {}
    for col in df_new.columns:
        features[f"{col}_mean"] = df_new[col].mean()
        features[f"{col}_var"] = df_new[col].var()
    features_df = pd.DataFrame([features])

    # --- Predict ---
    prediction = model.predict(features_df)[0]
    label = "Relaxed" if prediction == 0 else "Focused"

    print("Prediction for new EEG session:", label)

    # Wait 1 second before next simulated session
    time.sleep(1)
