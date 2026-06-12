"""
evaluate_metrics.py
--------------------
Calculates MAE, MSE, and R² for the linear regression predictions
two ways — manually with numpy and with sklearn — then prints them
side by side to verify they match.

Manual formulas:
  MAE  = mean(|actual - predicted|)
  MSE  = mean((actual - predicted)²)
  R²   = 1 - (SS_res / SS_tot)
           where SS_res = sum((actual - predicted)²)
                 SS_tot = sum((actual - mean(actual))²)

Input : regression_results_lat_temp.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Load results --------------------------------------------------------------

df = pd.read_csv("regression_results_lat_temp.csv")
print(f"Loaded {len(df):,} test set records.\n")

actual    = df["actual_temp_c"].values
predicted = df["predicted_temp_c"].values

# -- Manual calculations with numpy -------------------------------------------

# MAE: average of absolute differences
mae_manual = np.mean(np.abs(actual - predicted))

# MSE: average of squared differences
mse_manual = np.mean((actual - predicted) ** 2)

# R²: 1 minus ratio of residual variance to total variance
ss_res = np.sum((actual - predicted) ** 2)   # sum of squared residuals
ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # total sum of squares
r2_manual = 1 - (ss_res / ss_tot)

# -- Sklearn calculations ------------------------------------------------------

mae_sklearn = mean_absolute_error(actual, predicted)
mse_sklearn = mean_squared_error(actual, predicted)
r2_sklearn  = r2_score(actual, predicted)

# -- Print side by side --------------------------------------------------------

print("=" * 60)
print(f"{'Metric':<10} {'Manual (numpy)':>18} {'Sklearn':>18} {'Match?':>10}")
print("-" * 60)
print(f"{'MAE':<10} {mae_manual:>18.6f} {mae_sklearn:>18.6f} {' ✓' if round(mae_manual, 6) == round(mae_sklearn, 6) else ' ✗':>10}")
print(f"{'MSE':<10} {mse_manual:>18.6f} {mse_sklearn:>18.6f} {' ✓' if round(mse_manual, 6) == round(mse_sklearn, 6) else ' ✗':>10}")
print(f"{'R²':<10} {r2_manual:>18.6f} {r2_sklearn:>18.6f} {' ✓' if round(r2_manual, 6) == round(r2_sklearn, 6) else ' ✗':>10}")
print("=" * 60)

# -- Plain English interpretation ----------------------------------------------

print(f"""
Interpretation:
  MAE  = {mae_manual:.4f}°C  — on average, predictions are off by {mae_manual:.2f}°C
  MSE  = {mse_manual:.4f}    — average squared error (penalises large errors more)
  R²   = {r2_manual:.4f}     — latitude explains {r2_manual*100:.1f}% of the variance in temperature
""")
