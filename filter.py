from scipy.signal import butter, lfilter
import pandas as pd

# Function: Bandpass filter
def bandpass_filter(data, low=1, high=50, fs=250, order=5):
    nyq = 0.5 * fs
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(order, [low_cut, high_cut], btype='band')
    return lfilter(b, a, data)

# Load your simulated EEG
df = pd.read_csv("../data/simulated_eeg.csv")

# Apply filter to all channels
df_filtered = df.copy()
for col in df.columns:
    df_filtered[col] = bandpass_filter(df[col])

# Save the cleaned data
df_filtered.to_csv("../data/simulated_eeg_filtered.csv", index=False)
print("Data filtered and saved as simulated_eeg_filtered.csv!")
