import polars as pl
import datetime as dt
from wetterdienst.provider.dwd.observation import DwdObservationRequest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim.lr_scheduler as lr_scheduler

# 1. FETCH DATA
request = DwdObservationRequest(
    parameters=[
        ("daily", "kl"),                 # Climate (Temp, Humidity)
        ("daily", "precipitation_more"), # Rain and Snow
        ("daily", "solar")               # Sunshine
    ],
    start_date=dt.datetime(2004, 1, 1),
    end_date=dt.datetime(2026, 1, 2)
)

print("❄️ Fetching data for Rain and Snow prediction...")

# A. Interpolated Data (Regional)
values_interp = request.interpolate(latlon=(50.0, 8.9))
df_interp = values_interp.df.pivot(on="parameter", index="date", values="value", aggregate_function="first")

# B. Station Data (Local)
values_station = request.filter_by_station_id(station_id=["15555"]).values.all()
df_station = values_station.df.pivot(on="parameter", index="date", values="value", aggregate_function="first")

# C. Merge
df_merged = df_station.join(df_interp, on="date", how="inner", suffix="_interpolated")

# 2. FEATURE ENGINEERING
# If 'snow_depth_new' is missing, we create it as 0s so the code doesn't crash
if "snow_depth_new" not in df_merged.columns:
    df_merged = df_merged.with_columns(pl.lit(0.0).alias("snow_depth_new"))

df_ml = df_merged.with_columns([
    (pl.col("date").dt.month() * (2 * np.pi / 12)).sin().alias("month_sin"),
    (pl.col("date").dt.month() * (2 * np.pi / 12)).cos().alias("month_cos"),
    pl.col("precipitation_height").shift(1).alias("rain_yesterday"),
    pl.col("snow_depth_new").shift(1).alias("snow_yesterday"),
    # TARGETS
    pl.col("precipitation_height").shift(-1).alias("target_rain"),
    pl.col("snow_depth_new").shift(-1).alias("target_snow")
])

# 3. ROBUST FEATURE SELECTION
potential_features = [
    "month_sin", "month_cos", "rain_yesterday", "snow_yesterday",
    "humidity", "temperature_air_mean_2m", "sunshine_duration",
    "temperature_air_mean_2m_interpolated", "precipitation_height_interpolated"
]

# Identify columns that actually have data
usable_features = []
for f in potential_features:
    if f in df_ml.columns:
        # Check if the column is mostly empty (more than 90% nulls)
        if df_ml[f].is_null().mean() < 0.9:
            usable_features.append(f)

print(f"✅ Using {len(usable_features)} features: {usable_features}")

# 4. CLEANING (Handling Snow specifically)
# Fill snow nulls with 0 (DWD often leaves them null if there's no snow)
df_final = df_ml.with_columns([
    pl.col("target_snow").fill_null(0.0),
    pl.col("snow_yesterday").fill_null(0.0)
])

# Forward fill other sensors and drop remaining rows with missing targets
df_final = df_final.with_columns([
    pl.col(f).fill_null(strategy="forward") for f in usable_features
]).drop_nulls(subset=usable_features + ["target_rain"])

if df_final.is_empty():
    print("❌ ERROR: No rows remaining after cleaning. Check if date ranges overlap.")
    exit()

# 5. PREPARE AI DATA
X = df_final.select(usable_features).to_numpy()
y = df_final.select(["target_rain", "target_snow"]).to_numpy()

scaler_X = StandardScaler()
X_scaled = torch.FloatTensor(scaler_X.fit_transform(X))
y_tensor = torch.FloatTensor(y)



# 1. Convert the scaled X and the y tensor back to NumPy arrays
X_np = X_scaled.numpy()
y_np = y_tensor.numpy()

# 2. Combine them into one matrix (horizontal stack)
combined_data = np.hstack([X_np, y_np])

# 3. Create column names
# (Assumes 'usable_features' is a list of strings)
column_names = list(usable_features) + ["target_rain", "target_snow"]

# 4. Create DataFrame and Save
df_export = pd.DataFrame(combined_data, columns=column_names)
df_export.to_csv("prepared_ai_data.csv", index=False)

print("File saved successfully!")

# 6. MULTI-OUTPUT MODEL
class WeatherNet(nn.Module):
    def __init__(self, input_dim):
        super(WeatherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2) # Output 0: Rain, Output 1: Snow
        )
    def forward(self, x): return self.net(x)

model = WeatherNet(len(usable_features))
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print(f"🚀 Training on {len(df_final)} days...")

# 1. Setup your Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 2. Setup the Scheduler
# 'patience=10' means wait 10 epochs of no improvement before cutting LR
# 'factor=0.5' means cut the LR in half (0.005 -> 0.0025)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


# 1. Setup your Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 2. Setup the Scheduler
# 'patience=10' means wait 10 epochs of no improvement before cutting LR
# 'factor=0.5' means cut the LR in half (0.005 -> 0.0025)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

for epoch in range(501):
    model.train()
    optimizer.zero_grad()
    
    # Forward & Backward pass
    outputs = model(X_scaled)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    # 3. Step the scheduler! 
    # It needs to see the current loss to decide if it should trigger
    scheduler.step(loss)
    
    if epoch % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

# 7. PREDICTION FUNCTION
def predict_weather(date_str):
    try:
        target_dt = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except: return "Use YYYY-MM-DD"

    row = df_final.filter(pl.col("date") == target_dt)
    if row.is_empty():
        print(f"⚠️ No data available for {date_str}.")
        return

    inputs = torch.FloatTensor(scaler_X.transform(row.select(usable_features).to_numpy()))
    model.eval()
    with torch.no_grad():
        preds = model(inputs).numpy()[0]
        rain_p, snow_p = max(0, preds[0]), max(0, preds[1])
        act_r, act_s = row.select("target_rain")[0, 0], row.select("target_snow")[0, 0]

        print(f"\n📅 Forecast for the day after {date_str}:")
        print(f"RAIN: {rain_p:.2f} mm (Actual: {act_r:.2f} mm)")
        print(f"SNOW: {snow_p:.2f} cm (Actual: {act_s:.2f} cm)")

# TEST
predict_weather("2016-05-03")
predict_weather("2024-01-15")
predict_weather("2025-12-22")
predict_weather("2025-12-23")



predict_weather("2019-01-12")
predict_weather("2023-12-01")
predict_weather("2023-12-02")
predict_weather("2019-02-03")
predict_weather("2019-01-10")
predict_weather("2019-01-11")
predict_weather("2019-01-13")
predict_weather("2019-01-14")
predict_weather("2019-02-04")
predict_weather("2019-02-05")


predict_weather("2019-01-13")
predict_weather("2023-12-02")
predict_weather("2023-12-03")
predict_weather("2019-02-04")
predict_weather("2019-01-11")
predict_weather("2019-01-12")
predict_weather("2019-01-14")
predict_weather("2019-01-15")
predict_weather("2019-02-05")
predict_weather("2019-02-06")

df_final.write_csv("df_final_export.csv")
df_ml.write_csv("weather data_ml.csv")

predict_weather("2025-12-22")
predict_weather("2025-12-23")
predict_weather("2025-12-24")
predict_weather("2025-12-25")
predict_weather("2025-12-26")
predict_weather("2025-12-27")
predict_weather("2025-12-28")
predict_weather("2025-12-29")
predict_weather("2025-12-30")
predict_weather("2025-12-31")

# 8. EVALUATE MODEL
model.eval()
with torch.no_grad():
    predictions = model(X_scaled).numpy()
    # Ensure no negative predictions for physical reality
    predictions = np.maximum(0, predictions) 
    
    y_true = y_tensor.numpy()

# Calculate MAE (Average error in mm/cm)
mae_rain = mean_absolute_error(y_true[:, 0], predictions[:, 0])
mae_snow = mean_absolute_error(y_true[:, 1], predictions[:, 1])

# Calculate R2 (1.0 is a perfect fit, 0.0 is basically guessing)
r2_rain = r2_score(y_true[:, 0], predictions[:, 0])
r2_snow = r2_score(y_true[:, 1], predictions[:, 1])

print("\n--- MODEL ACCURACY REPORT ---")
print(f"RAIN Accuracy: Average error of ±{mae_rain:.2f} mm (R²: {r2_rain:.2f})")
print(f"SNOW Accuracy: Average error of ±{mae_snow:.2f} cm (R²: {r2_snow:.2f})")

if r2_rain > 0.1:
    print("⚠️ Warning: Rain prediction is weak. Consider adding more features or training longer.")

if r2_snow > 0.1:
    print("⚠️ Warning: Snow prediction is weak. Consider adding more features or training longer.")