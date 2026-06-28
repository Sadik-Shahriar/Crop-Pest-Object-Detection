# AgroPest-12: Image Dataset for Crop Pest Detection

## Overview

**AgroPest-12** is a curated image dataset containing 12 classes of crop insects and pests, designed for image classification and object detection tasks in agricultural AI research. Images are captured in natural agricultural environments, making the dataset suitable for training and evaluating deep learning models for real-world pest monitoring applications.

- **Dataset Name:** AgroPest-12: Image Dataset for Crop Pest Detection
- **Author:** Rupankar Majumdar
- **Platform:** Kaggle
- **Published:** 2025-09-17
- **License:** MIT
- **Tasks:** Image Classification, Object Detection

## Dataset Statistics

| Split          | Images    | Labels    |
|----------------|-----------|-----------|
| Training       | 11,502    | 11,502    |
| Validation     | 1,095     | 1,095     |
| Test           | 546       | 546       |
| **Total**      | **13,143**| **13,143**|

- **Total Size:** 537.38 MB (~0.52 GB)
- **Number of Classes:** 12
- **Image Format:** JPEG (.jpg)
- **Image Resolution:** 640x640 pixels (square)
- **Annotation Type:** Bounding boxes in YOLO format (normalized)
- **Annotation Format:** Ultralytics YOLO (.txt per image)

## Classes

The dataset contains the following 12 pest classes:

| Class ID | Name          |
|----------|---------------|
| 0        | Ants          |
| 1        | Bees          |
| 2        | Beetles       |
| 3        | Caterpillars  |
| 4        | Earthworms    |
| 5        | Earwigs       |
| 6        | Grasshoppers  |
| 7        | Moths         |
| 8        | Slugs         |
| 9        | Snails        |
| 10       | Wasps         |
| 11       | Weevils       |

## Data Format

### Directory Structure

The dataset follows the standard Ultralytics YOLO format:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### data.yaml (local)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 12
names:
  [
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
  ]
```

### data.yaml (Kaggle)

When using this dataset in a Kaggle notebook, update `data.yaml` to point to the Kaggle input path:

```yaml
train: /kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset/train/images
val: /kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset/valid/images
test: /kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset/test/images

nc: 12
names:
  [
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
  ]
```

### Annotation Format (YOLO)

Each image has a corresponding `.txt` file in the `labels/` directory. Each line in the annotation file represents one object:

```
<class_id> <center_x> <center_y> <width> <height>
```

Coordinates are normalized to [0, 1] relative to image width and height.

### Annotation Example

Label file content (`ants-1-_jpg.rf.8227b4d5f1cbbd72c290c1ca6012a337.txt`):

```
1 0.37421875 0.365625 0.31796875 0.5375
1 0.7265625 0.6203125 0.45234375 0.66875
```

This represents two bees (class_id=1) with bounding boxes in normalized coordinates.

## Notes

- The dataset was exported from Roboflow (filenames contain `.rf.` hashes).
- Each image has a matching label file (same name, .txt extension) in the corresponding `labels/` folder.
- Images are in `.jpg` format.
- Some classes may appear fewer times — check class distribution before training.
- The dataset paths in `data.yaml` use relative paths (`../train/images`) locally, but on Kaggle use absolute paths under `/kaggle/input/`.
- **Do not train on the raw dataset.** Use the cleaned version (see below).

## Label Quality Audit

A systematic audit comparing each file's **filename prefix** (the original Roboflow upload batch name, e.g. `ants-44-`) against the **class ID inside the `.txt` label file** revealed **36 mislabeled files (0.27%)** across the dataset.

The filename prefix reflects what the image actually contains — it was set by the person who collected and uploaded the images. The `.txt` label was assigned incorrectly by Roboflow at export time for these batches.

### Mismatch Summary

| Filename prefix implies | `.txt` label says | Count | Type |
|------------------------|-------------------|-------|------|
| Ants | Bees | 14 | Hard error — images show ants |
| Slugs | Snails | 6 | Ambiguous (similar species) |
| Caterpillars | Moths | 6 | Ambiguous (larva vs adult) |
| Bees | Beetles | 4 | Hard error — images show bees |
| Slugs | Ants | 3 | Hard error — nothing alike |
| Others | Mixed | 3 | Likely errors |

| Split | Files | Mismatches |
|-------|-------|------------|
| Train | 11,502 | 27 (0.23%) |
| Valid | 1,095  | 6  (0.55%) |
| Test  | 546    | 3  (0.55%) |
| **Total** | **13,143** | **36 (0.27%)** |

### Affected Batches

`ants-1`, `ants-43`, `ants-44`, `ants-442`, `ants-102`, `ants-432`,
`bees-103`, `bees-10`,
`slug-101`, `slug-109`, `slug-127`,
`catterpillar-247`, `catterpillar-292`, `catterpillar-297`, `catterpillar-298`, `catterpillar-353`,
`beetle-102`, `moth-1`

### Fix

The **EDA notebook (Notebook 1)** copies the raw dataset to `/kaggle/working/AgroPest-12-Cleaned/`, deletes the 36 mislabeled image+label pairs, and writes a corrected `data.yaml`. That output is published as a standalone Kaggle dataset named **AgroPest-12-Cleaned**.

- **Clean images**: 13,107 (removed 27 train + 6 valid + 3 test)
- **All training notebooks (2–5) use AgroPest-12-Cleaned as input, not the raw dataset**

## Potential Use Cases

- Drone-based pest monitoring
- Automated crop protection systems
- Early pest outbreak prediction
- Smart farming and precision agriculture
- Training and benchmarking image classification models (ResNet, EfficientNet, MobileNet, etc.)
- Training and benchmarking object detection models (YOLO, Faster R-CNN, RetinaNet, etc.)

## Citation

```
Rupankar Majumdar. "AgroPest-12: A 12-Class Image Dataset of Crop Insects and Pests."
Kaggle, 2025. https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset
```

## Source

- **Kaggle:** https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

## Paths

### Local (raw)

```
C:\Users\Sadik\Documents\Google Antigravity\CSE445\AgroPest-12 Image Dataset for Crop Pest Detection\
```

### Kaggle — Raw Dataset (DO NOT use for training)

```
/kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset/
├── data.yaml
├── train/   (11,502 images — 27 have wrong labels)
├── valid/   (1,095 images — 6 have wrong labels)
└── test/    (546 images — 3 have wrong labels)
```

### Kaggle — Clean Dataset (use for all training)

Dataset name: **AgroPest-12-Cleaned**  
Generated by: EDA notebook (Notebook 1), Section 8

```
/kaggle/input/datasets/sadikshahriar/agropest-12-cleaned/AgroPest-12-Cleaned/
├── data.yaml          ← point DATA_YAML here in all training notebooks
├── train/   (11,475 images — 27 removed)
├── valid/   (1,089 images — 6 removed)
└── test/    (543 images — 3 removed)
```
