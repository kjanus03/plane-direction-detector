import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import load_dataset

# === Load original dataset ===
ds = load_dataset("keremberke/plane-detection", name="full")

# === Load your manual annotations ===
with open("annotations.json") as f:
    manual_annotations = json.load(f)

# === Define augmentation pipeline ===
transform = A.Compose([
    A.Rotate(limit=30, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
    A.Normalize(),
    ToTensorV2()
])

# === Create output directory for augmented images ===
augmented_dir = "augmented_data"
os.makedirs(augmented_dir, exist_ok=True)

augmented_annotations = []

# === Generate augmented data ===
for entry in tqdm(ds['train']):
    image_id = str(entry['image_id'])
    if image_id not in manual_annotations:
        continue

    ann = manual_annotations[image_id]
    if not ann['plane_visible']:
        continue

    objects = entry['objects']
    if not objects['bbox']:
        continue

    # Get image and convert to array
    image_pil = entry['image']
    image_np = np.array(image_pil)

    # Crop the bounding box region
    bbox = objects['bbox'][0]  # x, y, w, h
    x, y, w, h = map(int, bbox)
    cropped = image_np[y:y+h, x:x+w]

    # Apply transformation
    augmented = transform(image=cropped)
    aug_img_tensor = augmented['image']
    aug_img_np = aug_img_tensor.permute(1, 2, 0).numpy()  # HWC

    # Save the augmented image
    save_path = os.path.join(augmented_dir, f"aug_{image_id}.png")
    Image.fromarray((aug_img_np * 255).astype(np.uint8)).save(save_path)

    # Store the annotation info
    augmented_annotations.append({
        "image_path": save_path,
        "image_id": int(image_id),
        "object_id": objects['id'][0],
        "bbox": bbox,
        "direction": ann['direction'],
        "contrail": ann['contrail'],
        "plane_visible": ann['plane_visible']
    })

# === Save updated annotations ===
with open("augmented_annotations.json", "w") as f:
    json.dump(augmented_annotations, f, indent=4)

print(f"Saved {len(augmented_annotations)} augmented samples to {augmented_dir}")
