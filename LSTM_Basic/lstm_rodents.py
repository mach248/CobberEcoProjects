"""
lstm_rodent_trends.py
---------------------
Predicts the next monthly capture-trend (-1, 0, +1) for each Portal plot
from a sliding window of the four previous trends, using an LSTM.

Pipeline:
  1. Load and merge the Portal survey datasets
  2. Filter to rodents and count captures per plot/year/month
  3. Encode month-over-month change as a trend class (-1, 0, +1)
  4. Build sliding-window samples (4 trends -> next trend)
  5. Train/test split, persistence baseline, then LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# ----------------------------
# Load the datasets
# ----------------------------
surveys = pd.read_csv("surveys.csv")
species = pd.read_csv("species.csv")
plots = pd.read_csv("plots.csv")

# ----------------------------
# Merge the datasets
# ----------------------------
data = pd.merge(surveys, species, on="species_id", how="left")
data = pd.merge(data, plots, on="plot_id", how="left")

# ----------------------------
# Filter to rodents only
# ----------------------------
rodents = data[data["taxa"] == "Rodent"].copy()

# ----------------------------
# Count rodent captures for each
# plot, year, and month
# ----------------------------
summary = (
    rodents
    .groupby(["plot_id", "year", "month"])
    .size()
    .reset_index(name="total_captures")
)

# ----------------------------
# Create a date column
# ----------------------------
summary["date"] = pd.to_datetime(
    dict(year=summary["year"],
         month=summary["month"],
         day=1)
)

# ----------------------------
# Sort chronologically
# ----------------------------
summary = summary.sort_values(
    by=["plot_id", "date"]
).reset_index(drop=True)

# ----------------------------
# Print first few rows
# ----------------------------
print(summary.head())

# ----------------------------
# Plot Plot #2
# ----------------------------
plot2 = summary[summary["plot_id"] == 2]

plt.figure(figsize=(10, 5))
plt.plot(
    plot2["date"],
    plot2["total_captures"],
    marker="o"
)

plt.title("Monthly Rodent Captures on Plot 2")
plt.xlabel("Date")
plt.ylabel("Total Rodent Captures")
plt.grid(True)

plt.tight_layout()
plt.savefig("rodent_timeseries.png", dpi=300)

plt.show()

# ----------------------------
# Compute change in captures
# within each plot
# ----------------------------
summary["capture_change"] = (
    summary.groupby("plot_id")["total_captures"]
           .diff()
)

# ----------------------------
# Encode the trend
# ----------------------------
def classify_trend(change):
    if change > 2:
        return 1
    elif change < -2:
        return -1
    else:
        return 0


summary["trend"] = summary["capture_change"].apply(
    lambda x: classify_trend(x) if pd.notna(x) else None
)

# ----------------------------
# Remove first observation from
# each plot (no previous month)
# ----------------------------
summary = summary.dropna(subset=["trend"]).reset_index(drop=True)

# Convert to integer values
summary["trend"] = summary["trend"].astype(int)

# ----------------------------
# Display the distribution
# ----------------------------
print("\nTrend value distribution:")
print(summary["trend"].value_counts().sort_index())

# ----------------------------
# Display the first few rows
# ----------------------------
print("\nFirst few rows:")
print(summary.head())

# -----------------------------------
# Create sliding windows
# -----------------------------------
window_size = 4

X = []
y = []

# Process each plot separately
for plot_id, group in summary.groupby("plot_id"):

    # Ensure observations are in time order
    group = group.sort_values("date")

    trends = group["trend"].to_numpy()

    # Skip plots with too few observations
    if len(trends) <= window_size:
        continue

    # Create windows
    for i in range(len(trends) - window_size):
        X.append(trends[i:i + window_size])
        y.append(trends[i + window_size])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# -----------------------------------
# Display results
# -----------------------------------
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("\nFirst five windows:\n")

for i in range(min(5, len(X))):
    print(f"Window {i+1}: {X[i]}  -->  Target: {y[i]}")

# -----------------------------------
# Split into training and test sets
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# -----------------------------------
# Display dataset sizes
# -----------------------------------
print("Training windows:", len(X_train))
print("Testing windows :", len(X_test))

print("\nShapes")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# -----------------------------------
# Persistence baseline
# -----------------------------------

# Predict the last trend in each window
y_pred_baseline = X_test[:, -1]

# Calculate accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"Persistence baseline accuracy: {baseline_accuracy:.3f}")
print(f"Persistence baseline accuracy: {baseline_accuracy * 100:.1f}%")

# ---------------------------------------
# Reshape for the LSTM
# (samples, time_steps, features)
# ---------------------------------------
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("X_train shape:", X_train_lstm.shape)
print("X_test shape :", X_test_lstm.shape)

# ---------------------------------------
# Build the LSTM model
# ---------------------------------------
model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(16),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# ---------------------------------------
# Train the model
# ---------------------------------------
history = model.fit(
    X_train_lstm,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# ---------------------------------------
# Make predictions
# ---------------------------------------
predictions = model.predict(X_test_lstm)

# Flatten from (N,1) to (N,)
predictions = predictions.flatten()

# ---------------------------------------
# Convert continuous predictions
# back to -1, 0, or 1
# ---------------------------------------
predicted_classes = np.where(
    predictions > 0.5,
    1,
    np.where(predictions < -0.5, -1, 0)
)

# ---------------------------------------
# Calculate accuracy
# ---------------------------------------
accuracy = accuracy_score(y_test, predicted_classes)

print(f"\nLSTM Accuracy: {accuracy:.3f}")
print(f"LSTM Accuracy: {accuracy * 100:.1f}%")

# ---------------------------------------
# Confusion matrix
# ---------------------------------------

# Define the class order explicitly so rows/columns are labeled predictably
labels = [-1, 0, 1]
class_names = ["Decline (-1)", "Stable (0)", "Increase (+1)"]

cm = confusion_matrix(y_test, predicted_classes, labels=labels)

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(cm)

# Per-class precision, recall, and F1
print("\nClassification report:")
print(classification_report(
    y_test,
    predicted_classes,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

# Plot it
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Reds", ax=ax, colorbar=True)

ax.set_title("LSTM Trend Prediction — Confusion Matrix")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png", dpi=300)
plt.show()

# ---------------------------------------
# Plot the training loss
# ---------------------------------------
plt.figure(figsize=(7, 5))

plt.plot(history.history["loss"], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("LSTM Training Loss")

plt.grid(True)

plt.tight_layout()
plt.savefig("lstm_loss.png", dpi=300)

plt.show()