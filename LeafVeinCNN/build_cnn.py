"""
build_cnn.py
------------
Builds, trains, and evaluates a small CNN on the leaf-vein images.

Pipeline:
  1. Load images, reshape to add a channel dimension -> (samples, 64, 64, 1)
  2. Build a small CNN (Conv -> ReLU -> MaxPool, twice, then Dense)
  3. Compile with Adam + sparse categorical crossentropy
  4. Train for 50 epochs (loss was still falling at 20, so we train longer)
  5. Plot training/validation accuracy and loss  -> training_curves.png
  6. Evaluate on the test set: print accuracy, build a confusion matrix
     -> confusion_matrix.png
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# -- Settings ------------------------------------------------------------------

classes  = ["pinnate", "palmate", "parallel"]   # label order: 0, 1, 2
IMG_SIZE = (64, 64)
EPOCHS   = 50

# -- Load data -----------------------------------------------------------------

images, labels = [], []
for label, cls in enumerate(classes):
    for p in sorted(glob.glob(os.path.join(cls, "*.png"))):
        arr = np.array(Image.open(p).convert("L").resize(IMG_SIZE), dtype="float32") / 255.0
        images.append(arr)
        labels.append(label)

X = np.array(images, dtype="float32")
y = np.array(labels, dtype="int64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -- 1. Reshape to add a channel dimension ------------------------------------

X_train = X_train.reshape(-1, 64, 64, 1)
X_test  = X_test.reshape(-1, 64, 64, 1)
print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

# -- 2. Build the CNN ----------------------------------------------------------

model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax"),
])
model.summary()

# -- 3. Compile ----------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -- 4. Train ------------------------------------------------------------------

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=2,
)

# -- 5. Training curves --------------------------------------------------------

fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

ax_acc.plot(history.history["accuracy"], label="Training", color="seagreen")
ax_acc.plot(history.history["val_accuracy"], label="Validation", color="tomato")
ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy over Epochs"); ax_acc.legend()
ax_acc.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

ax_loss.plot(history.history["loss"], label="Training", color="seagreen")
ax_loss.plot(history.history["val_loss"], label="Validation", color="tomato")
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss over Epochs"); ax_loss.legend()
ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nSaved: training_curves.png")

# -- 6. Evaluate on the test set ----------------------------------------------

# Predicted class = the output with the highest softmax probability
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {test_acc:.3f}  ({(y_test == y_pred).sum()}/{len(y_test)} correct)")

# Confusion matrix: rows = true class, columns = predicted class
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Greens")
fig.colorbar(im, ax=ax, label="Number of images")

# Tick labels with class names
ax.set_xticks(range(3)); ax.set_xticklabels(classes)
ax.set_yticks(range(3)); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted class", fontsize=12)
ax.set_ylabel("True class", fontsize=12)
ax.set_title(f"Confusion Matrix (test accuracy {test_acc:.2f})", fontsize=13)

# Write the count in each cell, in a contrasting color
thresh = cm.max() / 2 if cm.max() > 0 else 0.5
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")
plt.show()
