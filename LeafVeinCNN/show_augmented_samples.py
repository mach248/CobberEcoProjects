"""
show_augmented_samples.py
-------------------------
Loads 5 random augmented images from each leaf-vein class folder
(pinnate, palmate, parallel) and displays them in a 3 x 5 grid,
one class per row, labeled with the class name.

Output: augmented_samples.png
"""

import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image

# -- Settings ------------------------------------------------------------------

classes      = ["pinnate", "palmate", "parallel"]
N_PER_CLASS  = 5
SEED         = 42

random.seed(SEED)   # reproducible choice of which images are shown

# -- Build the grid ------------------------------------------------------------

fig, axes = plt.subplots(len(classes), N_PER_CLASS, figsize=(12, 7.5))

for row, cls in enumerate(classes):
    # Gather all PNGs in this class folder
    paths = glob.glob(os.path.join(cls, "*.png"))

    if len(paths) == 0:
        print(f"  WARNING: no PNG images found in folder '{cls}'")
        chosen = []
    else:
        # Pick 5 at random (or all of them if somehow fewer than 5)
        chosen = random.sample(paths, min(N_PER_CLASS, len(paths)))

    for col in range(N_PER_CLASS):
        ax = axes[row, col]
        ax.axis("off")
        if col < len(chosen):
            img = Image.open(chosen[col]).convert("L")
            ax.imshow(img, cmap="gray")
        # Label the first image in each row with the class name
        if col == 0:
            ax.set_title("")   # clear any default
            ax.text(-0.25, 0.5, cls, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", rotation=90,
                    va="center", ha="center")

fig.suptitle("Augmented Leaf-Vein Samples (5 per class)", fontsize=15)
plt.tight_layout()
plt.savefig("augmented_samples.png", dpi=150)
print("Saved: augmented_samples.png")
plt.show()
