# AgroPest-12 Part B: Software Libraries & Dependencies Guide

This document provides a comprehensive breakdown of the software libraries, packages, tools, and system dependencies utilized across all **Pretraining**, **Detection (Downstream Fine-Tuning & Baselines)**, and **Multi-Object Tracking** notebooks for Assignment Part B.

---

## 1. Quick Reference & Matrix of Dependencies

| Library / Package | Pretraining (`01a`–`04a`) | Detection (`00`, `01b`–`04b`) | Tracking (`05`) | Primary Functionality |
|---|:---:|:---:|:---:|---|
| **`torch` (PyTorch)** | **Core** | **Core** | **Core** | Standalone DDP dual-GPU engine, AMP, gradient clipping, custom SSL losses, weight surgery |
| **`torchvision`** | **Core** | — | — | SSL dual-view data augmentations (`RandomResizedCrop`, `ColorJitter`, `GaussianBlur`, `Solarize`) |
| **`ultralytics` (>=8.4.96)** | **Core** | **Core** | **Core** | YOLOv12 architecture parsing, YOLO detector fine-tuning, ByteTrack/BoT-SORT tracking engine |
| **`opencv-python` (`cv2`)** | — | **Core** | **Core** | Image decoding, bbox rendering, video stream handling (`VideoCapture`, `VideoWriter`), motion trails |
| **`scikit-learn`** | **Core** | Helper | Helper | t-SNE 2D projections, PCA, cosine similarity matrices, k-NN image retrieval |
| **`matplotlib` & `seaborn`** | **Core** | **Core** | **Core** | SSL loss curves, augmentation previews, PR curves, confusion matrices, telemetry dashboards |
| **`pandas` & `numpy`** | **Core** | **Core** | **Core** | Telemetry logging (`history.csv`), class-wise metric tables, frame-level tracking records |
| **`pyyaml` (`yaml`)** | Helper | **Core** | **Core** | Dynamic `data.yaml` generation and tracker configuration (`bytetrack.yaml`) |
| **`gdown`** | — | — | **Core** | Automated downloading of pest video datasets from Google Drive |
| **`ffmpeg`** (System utility) | — | — | **Core** | Transcoding OpenCV video outputs into web-standard H.264 (YUV420P) MP4 |
| **`kagglehub`** | Helper (`04a`) | — | — | Automated retrieval and caching of foundational vision model checkpoints |
| **`tqdm`** | — | Helper | **Core** | Visual progress tracking for frame-by-frame video processing |
| **`IPython.display`** | Helper | **Core** | **Core** | Rich formatting of metric comparison tables and inline HTML5 video player |

---

## 2. Pretraining Notebooks (`partb-01a` to `partb-04a`)

The pretraining phase implements a **traditional, self-contained PyTorch workflow** without third-party SSL frameworks (e.g., Lightly, PyTorch Lightning, or `ssldet`). Pretraining operates strictly on unlabelled images from `ssl_pool/images/` (9,180 images).

### Notebooks Covered:
1. `partb-01a-simclr-pretrain.ipynb` (SimCLR - Simple Framework for Contrastive Learning)
2. `partb-02a-byol-pretrain.ipynb` (BYOL - Bootstrap Your Own Latent)
3. `partb-03a-ijepa-pretrain.ipynb` (I-JEPA - Image-based Joint-Embedding Predictive Architecture)
4. `partb-04a-dinov3-pretrain.ipynb` (DINOv3 - Self-distillation with Vision Transformers)

### Libraries & Technical Roles:

#### 1. PyTorch (`torch`, `torch.nn`, `torch.distributed`, `torch.cuda.amp`)
* **Dual-GPU Distributed Training (DDP)**: Pretraining scripts are executed as standalone multi-process scripts using:
  ```bash
  python -m torch.distributed.run --standalone --nproc_per_node=2 ...
  ```
  `torch.distributed.init_process_group(backend="nccl")` coordinates inter-GPU communication across the two Kaggle Tesla T4 GPUs.
* **Automatic Mixed Precision (AMP)**: `torch.amp.autocast('cuda')` and `torch.cuda.amp.GradScaler()` reduce memory footprint and accelerate compute during contrastive representation learning.
* **Custom SSL Loss Objectives**:
  * **SimCLR**: Vectorized NT-Xent (Normalized Temperature-scaled Cross Entropy) with cosine similarity matrices, self-similarity masking, and temperature scaling ($\tau = 0.2$).
  * **BYOL**: Mean Squared Error (MSE) on $\ell_2$-normalized predictor outputs and EMA target projection, avoiding collapse without negative pairs.
  * **I-JEPA**: $\ell_1$ / $\ell_2$ smooth regression loss predicting target patch embeddings from masked context patch representations.
* **Teacher-Student EMA Updates**: Manual momentum parameter scheduling ($\tau = 0.996 \to 1.0$) for updating target networks without backpropagation (`torch.no_grad()`).

#### 2. TorchVision (`torchvision.transforms`, `transforms.v2`)
* **Stochastic Multi-View Data Augmentations**:
  * `transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0))`
  * `transforms.RandomHorizontalFlip(p=0.5)`
  * `transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)`
  * `transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))`
  * `transforms.RandomSolarize(threshold=128, p=0.2)` (crucial for SimCLR / BYOL view generation)
  * `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

#### 3. Ultralytics (`ultralytics`)
* **YOLOv12m Architecture Graph Extraction**:
  * Loads `yolo12m.yaml` configuration to instantiate a randomly initialized YOLOv12m network.
  * Extracts the convolutional backbone (layers 0 through 11) while preserving graph-routing metadata (`f`, `i`, and `save` attributes) required for detector neck integration.

#### 4. Scikit-Learn (`sklearn.manifold.TSNE`, `sklearn.decomposition.PCA`, `sklearn.metrics.pairwise`)
* **Unsupervised Representation Diagnostics**:
  * Computes 2D t-SNE embeddings of SSL feature representations extracted from a fixed 1,000-image SSL sample.
  * Computes cosine similarity matrices to retrieve top-5 nearest neighbors for query patches/images to verify feature clustering without reading ground truth labels.

#### 5. Matplotlib & Seaborn (`matplotlib.pyplot`, `seaborn`)
* Renders SSL loss curves across 30 pretraining epochs.
* Generates paired augmentation previews (View 1 vs. View 2).
* Visualizes t-SNE scatter plots and nearest-neighbor diagnostic grids.

#### 6. Pandas & NumPy (`pandas`, `numpy`)
* Logs epoch-by-epoch loss metrics to `history.csv`.
* Produces `run_manifest.json` and weight transfer reports.

---

## 3. Detection Notebooks (`partb-00` to `partb-04b`)

Downstream detection evaluates the effectiveness of the pretrained representations when fine-tuned on the standardized **20% labelled partition** (`finetune_20`, 2,295 images) for 50 epochs on a YOLOv12m detector.

### Notebooks Covered:
1. `partb-00-baselines-yolov12-rand20.ipynb` (Random Init Baseline)
2. `partb-00-baselines-yolov12-coco20.ipynb` (COCO-Pretrained Baseline)
3. `partb-01b-simclr-detect.ipynb` (SimCLR Pretrained Backbone + Fine-Tuning)
4. `partb-02b-byol-detect.ipynb` (BYOL Pretrained Backbone + Fine-Tuning)
5. `partb-03b-ijepa-detect.ipynb` (I-JEPA Pretrained Backbone + Fine-Tuning)
6. `partb-04b-dinov3-detect.ipynb` (DINOv3 Pretrained Backbone + Fine-Tuning)

### Libraries & Technical Roles:

#### 1. Ultralytics (`ultralytics.YOLO`)
* **Object Detection Engine**:
  * Initializes the YOLOv12m detector model (`yolo12m.yaml` or `yolo12m.pt`).
  * Runs fine-tuning over 50 epochs on the 20% labelled training split (`finetune_20/images/`).
  * Runs evaluation on `valid` (1,089 images) and `test` (543 images) partitions.
  * Generates core object detection metrics: $\text{mAP}_{50}$, $\text{mAP}_{50\text{-}95}$, Precision, Recall, and fitness scores.

#### 2. PyTorch (`torch`)
* **Backbone Weight Transfer & Verification (Weight Surgery)**:
  * Loads the pretrained `*_backbone.pt` checkpoint.
  * Injects pretrained weights into the detector's backbone layers (indices `0` to `11`).
  * Asserts 100% key and tensor shape compatibility before launching training.

#### 3. OpenCV (`cv2`)
* **Visual Evaluation & Error Analysis**:
  * Loads high-resolution test images.
  * Draws ground-truth vs. predicted bounding boxes with color-coded class designations.
  * Renders side-by-side comparative panels for error inspection (e.g., small pest detection, occlusion).

#### 4. PyYAML (`yaml`)
* **Dynamic Dataset Configuration**:
  * Programmatically writes `/kaggle/working/data.yaml` with absolute paths pointing to `finetune_20`, `valid`, and `test` splits of `AgroPest-12-PartB-Safe`.

#### 5. Matplotlib & Seaborn (`matplotlib.pyplot`, `seaborn`)
* Plots training loss trajectories (box loss, cls loss, dfl loss).
* Visualizes Precision-Recall (PR) curves, F1-confidence curves, and 12-class normalized confusion matrices.
* Generates comparative bar charts of per-class $\text{mAP}_{50\text{-}95}$.

#### 6. Pandas & NumPy (`pandas`, `numpy`)
* Parses validation and test metrics CSV files into structured DataFrames.
* Exports consolidated class-wise performance tables and final benchmark reports.

---

## 4. Multi-Object Tracking Notebook (`partb-05-tracking.ipynb`)

The tracking notebook evaluates the best-performing detector checkpoint across diverse video sequences (varying frame rates, resolutions, aspect ratios, and pest densities) using multi-object tracking (MOT) algorithms.

### Notebooks Covered:
1. `partb-05-tracking.ipynb` (Multi-Pest Tracking & Video Telemetry Dashboard)

### Libraries & Technical Roles:

#### 1. Ultralytics (`YOLO.track`, ByteTrack & BoT-SORT Modules)
* **Detection & Motion Association**:
  * Executes sequential inference on video frames.
  * Integrates the **ByteTrack** two-stage data association algorithm (associating high-confidence and low-confidence detection boxes using Kalman filter spatial predictions).
  * Automatically assigns and maintains consistent `track_id` values across temporal sequences.

#### 2. OpenCV (`cv2`)
* **Video Stream Decoding & Encoding**:
  * `cv2.VideoCapture`: Inspects video metadata (FPS, frame count, width, height, codec).
  * Frame processing: Resizes, crops, and normalizes frames for inference.
  * Visual annotation: Renders bounding boxes, class labels, tracking IDs, and historic centroid movement trails (polylines).
  * `cv2.VideoWriter`: Compiles processed annotated frames into video streams.

#### 3. GDown (`gdown`)
* **Automated Asset Retrieval**:
  * Automatically downloads raw `.mp4` video clips and archives from Google Drive directly into the Kaggle runtime environment.

#### 4. FFmpeg / Subprocess (`shutil.which("ffmpeg")`, `subprocess`)
* **Web-Standard Video Transcoding**:
  * Converts OpenCV default MP4 outputs (`mp4v` codec) into browser-compatible **H.264 (YUV420P)** format for display in Jupyter/Kaggle environments.

#### 5. TQDM (`tqdm`)
* Provides interactive progress bars for monitoring frame processing latency and throughput (FPS).

#### 6. Matplotlib & Seaborn (`matplotlib.pyplot`, `seaborn`)
* **4-Panel Diagnostic Telemetry Dashboards**:
  1. *Pest Count over Time*: Temporal count of active tracks per frame.
  2. *Spatial Trajectory Heatmap*: 2D kernel density estimation of pest movement coordinates.
  3. *Detection Confidence Distribution*: Track confidence histogram.
  4. *Track Lifespan / Duration Distribution*: Histogram of active frame count per unique ID.

#### 7. Pandas & NumPy (`pandas`, `numpy`)
* Aggregates frame-by-frame telemetry logs (`frame_idx`, `track_id`, `class_id`, `bbox`, `conf`).
* Calculates MOT metrics: total unique tracks, track fragmentation, and track switching statistics.

#### 8. IPython Display (`IPython.display.Video`, `IPython.display.HTML`)
* Embeds playable H.264 video players directly inside notebook output cells.

---

## 5. Pip Installation Commands Used in Notebooks

### In Pretraining Notebooks:
```bash
%pip install -q --upgrade "ultralytics>=8.4.96" "scikit-learn>=1.3" seaborn
```

### In Detection / Fine-Tuning Notebooks:
```bash
%pip install -q --upgrade "ultralytics>=8.4.96" "scikit-learn>=1.3" seaborn
```

### In Tracking Notebook:
```bash
%pip install -q gdown "ultralytics>=8.4.96" "scikit-learn>=1.3" seaborn
```

*(Note: PyTorch, TorchVision, and OpenCV are pre-installed in the Kaggle GPU environment and kept at their CUDA-matched versions).*
