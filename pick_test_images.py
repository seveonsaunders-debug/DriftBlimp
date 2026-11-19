"""
Copies training photos that actually contain a labeled object (such as glasses)
into a new folder, for use with fake_blimp_sender.py.

Run from the main DriftBlimp folder (the one that has the 'datasets' folder):
    python pick_test_images.py
Then send those photos with:
    python fake_blimp_sender.py test_images_glasses
"""
import shutil
from pathlib import Path

HOW_MANY = 10
MAX_BYTES = 2 * 1024 * 1024          # the receiver rejects photos bigger than 2 MB
OUT = Path("test_images_glasses")    # new folder; nothing existing is touched

datasets = Path("datasets")
if not datasets.is_dir():
    raise SystemExit("No 'datasets' folder here. Run this from the main DriftBlimp folder.")

OUT.mkdir(exist_ok=True)
copied = 0
for label_file in sorted(datasets.rglob("labels/*/*.txt")):
    if copied >= HOW_MANY:
        break
    if not label_file.read_text().strip():          # empty label = no object in photo
        continue
    # labels/<split>/<name>.txt  ->  images/<split>/<name>.jpg
    split_folder = label_file.parent.name
    image_folder = label_file.parent.parent.parent / "images" / split_folder
    for ext in (".jpg", ".jpeg"):
        image = image_folder / (label_file.stem + ext)
        if image.exists() and image.stat().st_size <= MAX_BYTES:
            shutil.copy(image, OUT / image.name)
            copied += 1
            break

print(f"Copied {copied} photos into {OUT}")
if copied == 0:
    print("Nothing found. Check that datasets\\<name>\\images and \\labels folders exist.")
