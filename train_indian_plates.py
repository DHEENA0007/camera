"""
Indian Number Plate Detection - Dataset Download & Model Training
================================================================
Downloads two Kaggle datasets, merges them into YOLO format,
and trains a YOLOv8s model for Indian license plate detection.

Usage:
    python train_indian_plates.py

Output:
    models/indian_plates_yolov8.pt  (best checkpoint)

Datasets used:
  1. dataclusterlabs/indian-number-plates-dataset
  2. kedarsai/indian-license-plates-with-labels
"""

import os
import sys
import glob
import shutil
import random
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# ─── Ensure dependencies are available ───────────────────────────────────────
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system(f"{sys.executable} -m pip install kagglehub -q")
    import kagglehub

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system(f"{sys.executable} -m pip install ultralytics -q")
    from ultralytics import YOLO

try:
    import yaml
except ImportError:
    os.system(f"{sys.executable} -m pip install pyyaml -q")
    import yaml

import cv2
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "indian_plates"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IMG = DATASET_DIR / "images" / "train"
VAL_IMG   = DATASET_DIR / "images" / "val"
TRAIN_LBL = DATASET_DIR / "labels" / "train"
VAL_LBL   = DATASET_DIR / "labels" / "val"

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Step 1: Download datasets ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Downloading datasets from Kaggle")
print("="*60)

print("\n[1/2] Downloading dataclusterlabs/indian-number-plates-dataset ...")
path1 = kagglehub.dataset_download("dataclusterlabs/indian-number-plates-dataset")
print(f"  -> {path1}")

print("\n[2/2] Downloading kedarsai/indian-license-plates-with-labels ...")
path2 = kagglehub.dataset_download("kedarsai/indian-license-plates-with-labels")
print(f"  -> {path2}")

# ─── Helper: Convert Pascal VOC XML annotation to YOLO format ────────────────
def voc_xml_to_yolo(xml_path, img_w, img_h):
    """Parse a VOC XML file and return list of YOLO bbox strings for class 0."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in [cx, cy, bw, bh]]
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines
    except Exception as e:
        print(f"  Warning: Failed to parse {xml_path}: {e}")
        return []


def coco_json_to_yolo(json_path, images_dir):
    """Parse COCO format JSON and return dict: image_filename -> [yolo_lines]."""
    with open(json_path) as f:
        coco = json.load(f)

    id_to_img = {img['id']: img for img in coco.get('images', [])}
    result = {}

    for ann in coco.get('annotations', []):
        img_info = id_to_img.get(ann['image_id'])
        if not img_info:
            continue
        fname = img_info['file_name']
        iw = img_info['width']
        ih = img_info['height']
        x, y, w, h = ann['bbox']  # COCO: x,y = top-left corner
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        bw = w / iw
        bh = h / ih
        cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in [cx, cy, bw, bh]]
        result.setdefault(fname, []).append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return result


def find_annotation_for_image(img_path, ann_dirs):
    """Find matching annotation file (XML or TXT) for an image."""
    stem = Path(img_path).stem
    for ann_dir in ann_dirs:
        for ext in ['.xml', '.txt']:
            candidate = Path(ann_dir) / (stem + ext)
            if candidate.exists():
                return str(candidate)
    return None


def get_image_size(img_path):
    """Return (w, h) of image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]


# ─── Step 2: Ingest datasets ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Ingesting & converting annotations")
print("="*60)

all_samples = []  # list of (img_path, [yolo_label_lines])

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def ingest_directory(base_path, source_name):
    """Recursively find all images + annotations in a downloaded dataset dir."""
    base = Path(base_path)
    print(f"\n  Scanning {source_name}: {base}")

    # ── Try COCO JSON first ──────────────────────────────────────────────
    json_files = list(base.rglob("*.json"))
    coco_maps = {}
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            if 'annotations' in data and 'images' in data:
                print(f"    Found COCO JSON: {jf}")
                coco_maps.update(coco_json_to_yolo(str(jf), str(jf.parent)))
        except Exception:
            pass

    # ── Find all images ──────────────────────────────────────────────────
    all_images = [p for p in base.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"    Found {len(all_images)} images")

    # Candidate annotation dirs: any folder containing XML or TXT annotation files
    ann_dirs = set()
    for ann_file in list(base.rglob("*.xml")) + list(base.rglob("*.txt")):
        # Skip YOLO dataset.yaml, README, etc.
        if ann_file.name.lower() in ['classes.txt', 'labels.txt']:
            continue
        ann_dirs.add(str(ann_file.parent))

    count = 0
    for img_path in all_images:
        yolo_lines = []

        # 1. COCO map
        for key in [img_path.name, str(img_path.relative_to(base))]:
            if key in coco_maps:
                yolo_lines = coco_maps[key]
                break

        # 2. Paired annotation file
        if not yolo_lines:
            ann_file = find_annotation_for_image(img_path, ann_dirs | {str(img_path.parent)})
            if ann_file:
                if ann_file.endswith('.xml'):
                    iw, ih = get_image_size(img_path)
                    if iw:
                        yolo_lines = voc_xml_to_yolo(ann_file, iw, ih)
                elif ann_file.endswith('.txt'):
                    # Already YOLO format — validate lines
                    raw = Path(ann_file).read_text().strip().splitlines()
                    validated = []
                    for line in raw:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                cls = int(parts[0])
                                coords = [float(x) for x in parts[1:]]
                                if all(0.0 <= c <= 1.0 for c in coords):
                                    # Force class to 0 (license_plate)
                                    validated.append(f"0 {' '.join(parts[1:])}")
                            except ValueError:
                                pass
                    yolo_lines = validated

        # Only include samples that have at least one valid annotation
        if yolo_lines:
            all_samples.append((str(img_path), yolo_lines))
            count += 1

    print(f"    Usable labeled samples: {count}")


ingest_directory(path1, "dataclusterlabs/indian-number-plates-dataset")
ingest_directory(path2, "kedarsai/indian-license-plates-with-labels")

print(f"\n  Total labeled samples collected: {len(all_samples)}")

if len(all_samples) == 0:
    print("\n[ERROR] No labeled samples found. Possible reasons:")
    print("  - Kaggle API credentials not configured")
    print("  - Dataset format not recognized")
    print("  - Try running: kaggle datasets download <dataset> manually")
    sys.exit(1)

# ─── Step 3: Split train/val ──────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Splitting into train / val sets (80/20)")
print("="*60)

random.seed(42)
random.shuffle(all_samples)
split = int(len(all_samples) * 0.8)
train_samples = all_samples[:split]
val_samples   = all_samples[split:]
print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}")

def copy_sample(img_path, yolo_lines, img_dir, lbl_dir, idx):
    """Copy image and write YOLO label file into dataset directory."""
    src = Path(img_path)
    ext = src.suffix.lower()
    # Use index-based name to avoid collisions from multiple source datasets
    stem = f"plate_{idx:06d}"
    dst_img = img_dir / (stem + ext)
    dst_lbl = lbl_dir / (stem + ".txt")
    shutil.copy2(str(src), str(dst_img))
    dst_lbl.write_text("\n".join(yolo_lines))

print("  Copying files...")
for i, (ip, ll) in enumerate(train_samples):
    copy_sample(ip, ll, TRAIN_IMG, TRAIN_LBL, i)
for i, (ip, ll) in enumerate(val_samples):
    copy_sample(ip, ll, VAL_IMG, VAL_LBL, len(train_samples) + i)

print("  Done.")

# ─── Step 4: Write dataset YAML ───────────────────────────────────────────────
dataset_yaml = DATASET_DIR / "dataset.yaml"
yaml_content = {
    'path': str(DATASET_DIR),
    'train': 'images/train',
    'val':   'images/val',
    'nc': 1,
    'names': ['license_plate'],
}
with open(dataset_yaml, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)
print(f"\n  Dataset YAML: {dataset_yaml}")

# ─── Step 5: Train YOLOv8 ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Training YOLOv8s on Indian License Plates")
print("="*60)

# Use YOLOv8s (small) - good balance of speed and accuracy
model = YOLO('yolov8s.pt')

results = model.train(
    data=str(dataset_yaml),
    epochs=50,
    imgsz=640,
    batch=16,
    name='indian_plates',
    project=str(MODELS_DIR / 'runs'),
    exist_ok=True,
    patience=15,          # Early stopping
    optimizer='AdamW',
    lr0=0.001,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,          # Small rotation (plates can be slightly tilted)
    translate=0.1,
    scale=0.4,
    shear=2.0,
    perspective=0.0002,
    flipud=0.0,           # Plates are never upside-down
    fliplr=0.0,           # Plates are not mirrored
    mosaic=0.8,
    mixup=0.0,
    copy_paste=0.0,
    conf=0.25,
    iou=0.5,
    verbose=True,
)

# ─── Step 6: Copy best weights to models/ ────────────────────────────────────
best_weights = MODELS_DIR / 'runs' / 'indian_plates' / 'weights' / 'best.pt'
output_model = MODELS_DIR / 'indian_plates_yolov8.pt'

if best_weights.exists():
    shutil.copy2(str(best_weights), str(output_model))
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved: {output_model}")
    print(f"{'='*60}")
else:
    print(f"\n[WARNING] best.pt not found at expected path: {best_weights}")
    # Search for it
    candidates = list((MODELS_DIR / 'runs').rglob('best.pt'))
    if candidates:
        shutil.copy2(str(candidates[0]), str(output_model))
        print(f"  Found and copied: {candidates[0]} -> {output_model}")

# ─── Step 7: Validate ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Validating trained model")
print("="*60)

if output_model.exists():
    trained = YOLO(str(output_model))
    metrics = trained.val(data=str(dataset_yaml), verbose=False)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"\nModel ready for use in the application.")
    print(f"Path: {output_model}")
else:
    print("  Model file not found — training may have failed.")
