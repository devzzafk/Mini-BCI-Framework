import numpy as np
import pandas as pd

# Simulate 5 seconds of EEG data at 250 Hz
fs = 250
time = np.arange(0, 5, 1/fs)
# Simulate 4 channels (like 4 brain sensors)
channels = 4
data = np.random.randn(len(time), channels)  # random signals

# Save as CSV
df = pd.DataFrame(data, columns=[f'Ch{i+1}' for i in range(channels)])
df.to_csv("data/simulated_eeg.csv", index=False)
print("Simulated EEG data saved!")
