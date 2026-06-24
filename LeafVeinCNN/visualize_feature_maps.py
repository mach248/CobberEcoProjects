"""
visualize_feature_maps.py
-------------------------
Opens the "black box" of the CNN by visualizing the feature maps produced
by the FIRST convolutional layer for a single test image.

A convolutional layer applies many small filters to the image; each filter
produces one "feature map" highlighting whatever that filter responds to
(an edge, a line at some angle, a vein junction). Seeing these maps shows
what low-level features the network extracts before making its decision.

This script trains the CNN, then:
  1. Builds an intermediate Keras Model: same input, output = first conv layer
  2. Runs one test image through it to get that layer's feature maps
  3. Displays all feature maps in a grid, titled with the image's true class
  4. Saves the figure as feature_maps.png
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# -- Settings ------------------------------------------------------------------

classes  = ["pinnate", "palmate", "parallel"]
IMG_SIZE = (64, 64)
EPOCHS   = 50
IMAGE_INDEX = 0   # which test image to visualize (change to explore others)

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
X_train = X_train.reshape(-1, 64, 64, 1)
X_test  = X_test.reshape(-1, 64, 64, 1)

# -- Build and train the CNN ---------------------------------------------------

model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(16, (3, 3), activation="relu", name="conv1"),   # named for easy access
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu", name="conv2"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=EPOCHS,
          validation_data=(X_test, y_test), verbose=0)
print("Training complete.")

# -- 1. Build an intermediate model: input -> first conv layer output ---------
# Re-apply the trained layers to a fresh Input, stopping at conv1. This reuses
# the already-trained weights and works reliably across Keras versions.

inp = Input(shape=(64, 64, 1))
x = inp
for layer in model.layers:
    x = layer(x)
    if layer.name == "conv1":
        break
feature_model = Model(inputs=inp, outputs=x)

# -- 2. Run ONE test image through it -----------------------------------------

one_image = X_test[IMAGE_INDEX:IMAGE_INDEX + 1]        # keep batch dim: (1,64,64,1)
feature_maps = feature_model.predict(one_image)         # shape: (1, H, W, n_filters)
feature_maps = feature_maps[0]                          # drop batch -> (H, W, n_filters)

true_class = classes[y_test[IMAGE_INDEX]]
n_filters  = feature_maps.shape[-1]
print(f"Visualizing {n_filters} feature maps for a '{true_class}' image.")

# -- 3. Display all feature maps in a grid ------------------------------------

cols = 4
rows = int(np.ceil(n_filters / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
axes = np.array(axes).ravel()

for k in range(len(axes)):
    ax = axes[k]
    ax.axis("off")
    if k < n_filters:
        ax.imshow(feature_maps[:, :, k], cmap="viridis")
        ax.set_title(f"Filter {k}", fontsize=9)

fig.suptitle(f"First Conv Layer Feature Maps\n(true class: {true_class})", fontsize=14)
plt.tight_layout()
plt.savefig("feature_maps.png", dpi=150)
print("Saved: feature_maps.png")
plt.show()
