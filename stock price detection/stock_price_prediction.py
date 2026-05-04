# ============================================================
# Stock Price Prediction using Linear Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("stock_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print("Dataset Shape:", df.shape)
print(df.head())

# ── 2. Feature Engineering ───────────────────────────────────
# Use a numerical day-index as the predictor (simple univariate regression)
df["Day"] = np.arange(len(df))

# Additional features: Open, High, Low, Volume → predict Close
feature_cols = ["Day", "Open", "High", "Low", "Volume"]
target_col   = "Close"

X = df[feature_cols].values
y = df[target_col].values

# ── 3. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False   # time-series: no shuffle
)

# ── 4. Feature Scaling ───────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Train Linear Regression Model ─────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name:10s}: {coef:.4f}")
print(f"  Intercept  : {model.intercept_:.4f}")

# ── 6. Predictions ───────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# ── 7. Evaluation Metrics ────────────────────────────────────
def print_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

print_metrics(y_train, y_pred_train, "── Training Set Metrics ──")
print_metrics(y_test,  y_pred_test,  "── Test Set Metrics ──")

# ── 8. Visualisations ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Stock Price Prediction — Linear Regression", fontsize=16, fontweight="bold")

dates      = df["Date"].values
n_train    = len(y_train)
train_dates = dates[:n_train]
test_dates  = dates[n_train:]

# ── Plot 1: Full Actual vs Predicted ─────────────────────────
ax1 = axes[0, 0]
ax1.plot(train_dates, y_train,      label="Train Actual",    color="#2196F3", linewidth=1.5)
ax1.plot(train_dates, y_pred_train, label="Train Predicted", color="#4CAF50", linestyle="--", linewidth=1.5)
ax1.plot(test_dates,  y_test,       label="Test Actual",     color="#FF5722", linewidth=1.5)
ax1.plot(test_dates,  y_pred_test,  label="Test Predicted",  color="#FF9800", linestyle="--", linewidth=1.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.legend(fontsize=8)
ax1.set_title("Actual vs Predicted Close Price")
ax1.set_ylabel("Price (USD)")
ax1.grid(True, alpha=0.3)

# ── Plot 2: Test set zoom ─────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(test_dates, y_test,      label="Actual",    color="#FF5722", linewidth=2)
ax2.plot(test_dates, y_pred_test, label="Predicted", color="#FF9800", linestyle="--", linewidth=2, marker="o", markersize=4)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax2.legend()
ax2.set_title("Test Set: Actual vs Predicted")
ax2.set_ylabel("Price (USD)")
ax2.grid(True, alpha=0.3)

# ── Plot 3: Residuals ─────────────────────────────────────────
ax3 = axes[1, 0]
residuals = y_test - y_pred_test
ax3.bar(range(len(residuals)), residuals, color=["#4CAF50" if r >= 0 else "#F44336" for r in residuals])
ax3.axhline(0, color="black", linewidth=1)
ax3.set_title("Residuals (Test Set)")
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Residual (USD)")
ax3.grid(True, alpha=0.3)

# ── Plot 4: Actual vs Predicted Scatter ───────────────────────
ax4 = axes[1, 1]
ax4.scatter(y_test, y_pred_test, color="#9C27B0", alpha=0.7, edgecolors="white", s=60)
min_val = min(y_test.min(), y_pred_test.min()) - 5
max_val = max(y_test.max(), y_pred_test.max()) + 5
ax4.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
ax4.set_title("Actual vs Predicted (Scatter)")
ax4.set_xlabel("Actual Price (USD)")
ax4.set_ylabel("Predicted Price (USD)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stock_prediction_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved as stock_prediction_results.png")
plt.show()

# ── 9. Predict a Future Price ─────────────────────────────────
# Simulate the "next day" beyond the dataset
last_row    = df.iloc[-1]
next_day    = last_row["Day"] + 1
next_open   = last_row["Close"]       # assume open ≈ previous close
next_high   = next_open * 1.01
next_low    = next_open * 0.99
next_volume = last_row["Volume"]

future_input = np.array([[next_day, next_open, next_high, next_low, next_volume]])
future_scaled = scaler.transform(future_input)
future_pred   = model.predict(future_scaled)[0]

print(f"\nFuture Price Prediction (Day {int(next_day)}):")
print(f"  Estimated Close Price: ${future_pred:.2f}")
