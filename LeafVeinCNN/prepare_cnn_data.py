"""
prepare_cnn_data.py
-------------------
Loads all leaf-vein images, preprocesses them, and splits into
training and test sets ready for a CNN.

Steps:
  1. Load every PNG from the three class folders
  2. Convert to grayscale, resize to 64x64
  3. Normalize pixel values to 0-1 (divide by 255)
  4. Build X (images) and y (integer labels: pinnate=0, palmate=1, parallel=2)
  5. Split 80/20 into train/test (random_state=42)
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# -- Settings ------------------------------------------------------------------

# Order defines the integer label: pinnate=0, palmate=1, parallel=2
classes   = ["pinnate", "palmate", "parallel"]
IMG_SIZE  = (64, 64)

# -- Load and preprocess every image ------------------------------------------

images = []
labels = []

for label, cls in enumerate(classes):
    paths = sorted(glob.glob(os.path.join(cls, "*.png")))
    print(f"{cls} (label {label}): found {len(paths)} images")

    for p in paths:
        img = Image.open(p).convert("L").resize(IMG_SIZE)   # grayscale + resize
        arr = np.array(img, dtype="float32") / 255.0         # normalize to 0-1
        images.append(arr)
        labels.append(label)

# -- Build numpy arrays --------------------------------------------------------

X = np.array(images, dtype="float32")   # shape: (n_images, 64, 64)
y = np.array(labels, dtype="int64")     # shape: (n_images,)

print(f"\nFull dataset: X shape {X.shape}, y shape {y.shape}")

# -- Train/test split (80/20) --------------------------------------------------

# stratify=y keeps the class proportions balanced across train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nAfter 80/20 split:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test : {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test : {y_test.shape}")
