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

| Split          | Images    |
|----------------|-----------|
| Training       | 11,502    |
| Validation     | 1,095     |
| Test           | 546       |
| **Total**      | **13,143**|

- **Total Size:** < 600 MB
- **Number of Classes:** 12
- **Annotation Type:** Bounding boxes in YOLO format

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

### data.yaml

```yaml
train: /path/to/train/images
val: /path/to/valid/images
test: /path/to/test/images

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
