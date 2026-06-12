"""
linear_regression_lat_temp.py
------------------------------
Fits a simple linear regression model using scikit-learn to predict
mean annual temperature (BIO1) from latitude for red fox (Vulpes vulpes)
GBIF occurrence records.

Workflow:
  1. Load enriched CSV and drop rows with missing values
  2. Split into 80% training and 20% test sets (random_state=42)
  3. Fit a LinearRegression model on the training set
  4. Generate predictions on the test set
  5. Save actual values, predicted values, and residuals to a DataFrame
  6. Print model coefficients, performance metrics, and a preview of results

Input  : vulpes_vulpes_red_fox_GBIF_with_climate.csv
Output : regression_results_lat_temp.csv
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# -- Configuration -------------------------------------------------------------

INPUT_CSV  = "vulpes_vulpes_red_fox_GBIF_with_climate.csv"
OUTPUT_CSV = "regression_results_lat_temp.csv"

PREDICTOR = "latitude"
OUTCOME   = "mean_annual_temp_c"

# -- Load and clean ------------------------------------------------------------

print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  Records loaded     : {len(df):,}")

df = df.dropna(subset=[PREDICTOR, OUTCOME])
print(f"  After dropping NaN : {len(df):,}")

# -- Features and target -------------------------------------------------------

X = df[[PREDICTOR]]   # 2D array required by sklearn
y = df[OUTCOME]

# -- Train/test split ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Training rows : {len(X_train):,}")
print(f"  Test rows     : {len(X_test):,}")

# -- Fit model -----------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

slope     = model.coef_[0]
intercept = model.intercept_

print(f"\nModel coefficients:")
print(f"  Slope (°C per degree latitude) : {slope:.4f}")
print(f"  Intercept                      : {intercept:.4f}")
print(f"\n  Interpretation: for every 1° increase in latitude,")
print(f"  temperature changes by {slope:.4f}°C")

# -- Predictions and residuals -------------------------------------------------

y_pred     = model.predict(X_test)
residuals  = y_test.values - y_pred

results = pd.DataFrame({
    "actual_temp_c"    : y_test.values,
    "predicted_temp_c" : y_pred.round(2),
    "residual"         : residuals.round(2),
})
results = results.reset_index(drop=True)

# -- Performance metrics -------------------------------------------------------

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel performance on test set:")
print(f"  R²   : {r2:.4f}  (proportion of variance explained)")
print(f"  MAE  : {mae:.4f}°C  (mean absolute error)")
print(f"  RMSE : {rmse:.4f}°C  (root mean squared error)")

# -- Preview results -----------------------------------------------------------

print(f"\nFirst 10 rows of results:")
print("-" * 45)
print(results.head(10).to_string(index=True))

# -- Save ----------------------------------------------------------------------

results.to_csv(OUTPUT_CSV, index=False)
print(f"\nFull results saved: {OUTPUT_CSV}")
