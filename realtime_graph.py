import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("../models/eeg_model.pkl")

fs = 250  # samples per second
seconds = 5
channels = 4

# Store history for plotting
history = []

plt.ion()  # interactive mode on
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-')
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("Time step")
ax.set_ylabel("Mental state (0=Relaxed,1=Focused)")
ax.set_title("Real-Time EEG Predictions")

print("Starting real-time EEG predictions with graph (Ctrl+C to stop)...")

step = 0
try:
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
        history.append(prediction)
        step += 1

        print(f"Step {step}: Prediction = {label}")

        # --- Update plot ---
        line.set_xdata(range(len(history)))
        line.set_ydata(history)
        ax.set_xlim(0, len(history)+1)
        plt.pause(0.1)

        time.sleep(0.5)  # wait half a second for next session

except KeyboardInterrupt:
    print("\nStopped by user.")
    plt.ioff()
    plt.show()
