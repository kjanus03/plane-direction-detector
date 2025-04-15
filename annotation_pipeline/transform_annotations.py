import json

with open("annotations.json") as f:
    manual_annotations = json.load(f)

from datasets import load_dataset
ds = load_dataset("keremberke/plane-detection", name="full")

formatted_annotations = []


for entry in ds['train']:
    image_id = str(entry['image_id'])
    if image_id in manual_annotations:
        ann = manual_annotations[image_id]

        objects = entry['objects']
        if objects['bbox']:
            formatted_annotations.append({
                "image_id": entry['image_id'],
                "object_id": objects['id'][0],
                "bbox": objects['bbox'][0],
                "direction": ann['direction'],
                "contrail": ann['contrail'],
                "plane_visible": ann['plane_visible']
            })

with open("formatted_annotations.json", "w") as f:
    json.dump(formatted_annotations, f, indent=4)

print(f"Saved {len(formatted_annotations)} formatted annotations.")
