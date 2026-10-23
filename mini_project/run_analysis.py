"""Reproduce the BDML urban water-demand forecasting experiment."""
from pathlib import Path
import csv, math
import numpy as np

DATA = Path(__file__).parent / "data" / "urban_water_demand.csv"
rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
X, y = [], []
for i, row in enumerate(rows):
    from datetime import date
    day = date.fromisoformat(row["date"]); doy = day.timetuple().tm_yday
    X.append([float(row["temperature_c"]), float(row["rainfall_mm"]), float(row["humidity_pct"]),
              float(row["is_weekend"]), float(row["is_holiday"]), i,
              math.sin(2*math.pi*doy/365.25), math.cos(2*math.pi*doy/365.25)])
    y.append(float(row["demand_litres"]))
X, y = np.asarray(X), np.asarray(y)
split = int(.8*len(y))
mean, scale = X[:split].mean(0), X[:split].std(0); scale[scale == 0] = 1
Z = np.column_stack([np.ones(len(y)), (X-mean)/scale])
coef = np.linalg.lstsq(Z[:split], y[:split], rcond=None)[0]
prediction, truth = Z[split:] @ coef, y[split:]
mae = np.mean(np.abs(truth-prediction))
rmse = np.sqrt(np.mean((truth-prediction)**2))
r2 = 1-np.sum((truth-prediction)**2)/np.sum((truth-truth.mean())**2)
baseline_mae = np.mean(np.abs(truth-y[:split].mean()))
print(f"Test MAE: {mae:.2f} L/day")
print(f"Test RMSE: {rmse:.2f} L/day")
print(f"Test R2: {r2:.4f}")
print(f"Mean-baseline MAE: {baseline_mae:.2f} L/day")
