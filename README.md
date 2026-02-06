🌌 About the Project

This repository contains a Mini Brain–Computer Interface (BCI) Framework that simulates EEG brain signals, processes them, and uses a trained machine learning model to make real-time predictions — all visualized through a Streamlit web app.

Since access to real Neuralink or medical-grade EEG devices is not possible, I created simulated EEG data to demonstrate how a BCI pipeline works in practice — from raw signals → filtering → feature extraction → prediction → visualization.

This project is part of my learning journey in AI, data science, and neurotechnology, and is showcased on my YouTube channel Juneverse.

✨ Features

✅ Simulated EEG signal generation

✅ Noise filtering of brain signals

✅ Feature extraction from EEG data

✅ Machine learning model training

✅ Real-time prediction system

✅ Live EEG visualization

✅ Interactive Streamlit dashboard

🛠️ Tech Stack

Language: Python

Libraries:

Streamlit

NumPy

Pandas

Matplotlib

Scikit-learn

Tools:

Git & GitHub

VS Code

Jupyter Notebook

📂 Repository Structure
README.md                 → Project overview  
simulate_eeg.py           → Generates fake EEG signals  
filter.py                 → Cleans and filters EEG data  
features.py               → Extracts useful features  
train_model.py            → Trains ML model  
eeg_model.pkl             → Saved trained model  
predict_eeg.py            → Uses model to predict  
realtime_graph.py         → Live brain signal plots  
realtime_predict.py       → Real-time predictions  
streamlit_app.py          → Main web dashboard  
requirements.txt          → Required libraries  
features.csv              → Extracted features dataset  
simulated_eeg_filtered.csv → Filtered EEG data  
🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/mini-bci-framework.git

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run streamlit_app.py
🎯 Goal of This Project

My goal was to:

Understand how BCIs work

Build a mini version of Neuralink-style brain data processing

Learn ML + real-time dashboards

Make complex neurotech concepts simple and visual

🎬 YouTube

I explain this project step-by-step on my channel Juneverse — where I document my coding journey, creative tech experiments, and learning process as a student developer.

🔗 https://youtu.be/umKl40rpDaY

👩‍💻 Author

Devi Chandran .S
Aspiring AI + Cloud Engineer | Student Developer | Tech Creator

⭐ If you like this project…

Feel free to:

Star this repo ⭐

Fork it 🍴

Experiment with it 🚀

Or suggest improvements!
