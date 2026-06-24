"""
classify_new_leaf.py
--------------------
The true generalization test: classify a brand-new leaf drawing
(test_image.png) that was never part of training or augmentation.

Steps:
  1. Train the CNN on the augmented dataset (as before)
  2. Load test_image.png -> grayscale -> 64x64 -> normalize -> reshape
  3. Run it through the model
  4. Print the predicted class and the confidence for all three classes
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# -- Settings ------------------------------------------------------------------

classes    = ["pinnate", "palmate", "parallel"]
IMG_SIZE   = (64, 64)
EPOCHS     = 50
TEST_IMAGE = "test_image.png"

# -- Load training data and train ---------------------------------------------

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
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=EPOCHS,
          validation_data=(X_test, y_test), verbose=0)
print("Training complete.\n")

# -- Load and preprocess the new image ----------------------------------------

if not os.path.exists(TEST_IMAGE):
    raise FileNotFoundError(
        f"Could not find '{TEST_IMAGE}'. Place your new drawing in this folder."
    )

# Flatten any transparency onto white first (same fix as augmentation),
# so a transparent background does not become black.
raw = Image.open(TEST_IMAGE)
if raw.mode in ("RGBA", "LA", "P"):
    raw = raw.convert("RGBA")
    bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    raw = Image.alpha_composite(bg, raw)

img = raw.convert("L").resize(IMG_SIZE)               # grayscale + resize
arr = np.array(img, dtype="float32") / 255.0          # normalize 0-1
arr = arr.reshape(1, 64, 64, 1)                       # add batch + channel dims

# -- Predict -------------------------------------------------------------------

probs = model.predict(arr, verbose=0)[0]              # 3 softmax probabilities
predicted_label = int(np.argmax(probs))
predicted_class = classes[predicted_label]

print(f"Predicted class: {predicted_class}\n")
print("Confidence scores:")
print("-" * 32)
for cls, p in zip(classes, probs):
    bar = "#" * int(round(p * 20))                    # little text bar
    marker = "  <-- predicted" if cls == predicted_class else ""
    print(f"  {cls:<10} {p*100:5.1f}%  {bar}{marker}")
