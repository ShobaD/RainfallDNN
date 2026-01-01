🌦️ DeepWeather: Multi-Output Rainfall & Snow Prediction
This project implements a Deep Neural Network (DNN) to predict the next day's precipitation and snow depth. It leverages the DWD (Deutscher Wetterdienst) API for high-quality station and interpolated regional data, utilizing a multi-output architecture to solve for two distinct targets simultaneously.

🧠 The Philosophy: Weilheim vs. Chennai
This model was built to explore the difference between seasonal, predictable weather (Weilheim, Germany) and extreme, high-variance weather (Chennai, India).

The Engineer's Question: Can we predict the exact rainfall amount?

The Product Manager's Question: Will it rain today? If so, will it be manageable or chaotic?

🛠️ Tech Stack
Data Processing: Polars (Blazingly fast DataFrame operations)

API Ingestion: Wetterdienst (German Weather Service wrapper)

AI Engine: PyTorch (Sequential DNN with ReLU activation)

Preprocessing: Scikit-Learn (StandardScaler for feature normalization)

🏗️ Model Architecture
The network is a 3-layer fully connected architecture designed for Multi-Output Regression:

Input Layer: Dynamic (based on usable features)

Hidden Layer 1: 64 Neurons + ReLU

Hidden Layer 2: 32 Neurons + ReLU

Output Layer: 2 Neurons (Linear) -> [Rainfall (mm), Snow Depth (cm)]

🚀 Key Features
Cyclical Encoding: Months are encoded using Sine and Cosine transformations to capture the circular nature of seasons.

Hybrid Data: Merges local station observations with regional interpolated data for increased robustness.

Lag Features: Incorporates "yesterday's" weather to provide the model with short-term memory.

Auto-Cleaning: Automatically identifies and drops sensors with >90% missing data.

