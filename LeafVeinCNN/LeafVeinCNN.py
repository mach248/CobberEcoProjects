"""
augment_leaves.py
-----------------
Creates 40 augmented versions of each original leaf-vein drawing,
giving 120 training images total (40 x 3 classes) for a CNN.

Robust handling of two common problems:
  1. Black rotation corners -- fixed by flattening any transparency onto a
     white background first, then rotating with a white fill color.
  2. Leftover files from earlier runs -- old aug_*.png files are cleared
     from each class folder before new ones are generated.

Augmentations applied (random combination per image):
  - rotation (small angle)
  - horizontal flip
  - light blur
Vertical flips are OFF by default (a top-to-bottom leaf looks unnatural);
set ALLOW_VERTICAL_FLIP = True to enable them.

Folder layout expected (originals already in place):
  pinnate/pinnate_1.png
  palmate/palmate_1.png
  parallel/parallel_1.png
"""

import os
import glob
import random
from PIL import Image, ImageFilter

# -- Settings ------------------------------------------------------------------

N_PER_CLASS         = 40
IMAGE_SIZE          = (128, 128)
ALLOW_VERTICAL_FLIP = False
SEED                = 42

classes = ["pinnate", "palmate", "parallel"]
originals = [
    "pinnate/pinnate_1.png",
    "palmate/palmate_1.png",
    "parallel/parallel_1.png",
]

random.seed(SEED)   # seed ONCE for reproducible runs with real variety


def load_on_white(path, size):
    """Open an image, flatten any transparency onto white, return grayscale.

    This prevents transparent regions from turning black when converted to
    grayscale -- the root cause of black corners after rotation.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        white_bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white_bg, im)
    return im.convert("L").resize(size)


def clear_old_augmented(folder):
    """Delete any existing aug_*.png files so we start clean."""
    for old in glob.glob(os.path.join(folder, "aug_*.png")):
        os.remove(old)


def augment_image(base_img, output_path, n):
    """Save n randomly augmented copies of base_img into output_path."""
    os.makedirs(output_path, exist_ok=True)
    clear_old_augmented(output_path)

    for i in range(n):
        aug = base_img.copy()

        # Rotation, filling the new corners with white (255).
        # resample=BICUBIC keeps the rotated veins smooth.
        angle = random.uniform(-30, 30)
        aug = aug.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

        # Horizontal flip ~50% of the time
        if random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Optional vertical flip
        if ALLOW_VERTICAL_FLIP and random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_TOP_BOTTOM)

        # Light blur ~30% of the time
        if random.random() > 0.7:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))

        aug.save(os.path.join(output_path, f"aug_{i:03d}.png"))


# -- Run -----------------------------------------------------------------------

total = 0
for cls, orig_path in zip(classes, originals):
    if not os.path.exists(orig_path):
        print(f"  WARNING: could not find {orig_path} -- skipping {cls}")
        continue
    base = load_on_white(orig_path, IMAGE_SIZE)
    augment_image(base, cls, n=N_PER_CLASS)
    total += N_PER_CLASS
    print(f"Augmented {cls}: {N_PER_CLASS} images saved")

print(f"\nDone. {total} augmented images created "
      f"({N_PER_CLASS} per class x {len(classes)} classes).")