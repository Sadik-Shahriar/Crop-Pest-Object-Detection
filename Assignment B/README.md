# CSE 445 — Assignment B: Self-Supervised Pretraining for Label-Efficient Object Detection

## Group Info

- **Student:** Md. Sadik Shahriar | **ID:** 2023-2-60-103
- **Dataset:** AgroPest-12-Cleaned (13,107 images, 12 pest classes, 640x640)
- **Best Part-A Model:** YOLOv12-m (test mAP50-95 = 0.477)
- **Platform:** Kaggle — 2x Tesla T4 GPUs
- **Deadline:** August 16, 2026

---

## 1. Part B Overview

Replace the randomly-initialised/COCO-pretrained backbone of your best Part-A YOLO with backbones pretrained by **4 SSL methods** (SimCLR, BYOL, I-JEPA, DINOv3). Fine-tune each at a **20% label budget** under a leakage-safe 10/10/80 data partition. Deploy the best detector in a tracking pipeline on video.

### Deliverables (9 notebooks + 1 bonus + report)

| Notebook | Kaggle Title | What It Does |
|----------|-------------|--------------|
| 1a | `groupXX-partB-01a-simclr-pretrain` | SimCLR pretraining on 80% unlabelled pool |
| 1b | `groupXX-partB-01b-simclr-detect` | SimCLR backbone → YOLO fine-tune @ 20% |
| 2a | `groupXX-partB-02a-byol-pretrain` | BYOL pretraining |
| 2b | `groupXX-partB-02b-byol-detect` | BYOL backbone → YOLO fine-tune @ 20% |
| 3a | `groupXX-partB-03a-ijepa-pretrain` | I-JEPA pretraining |
| 3b | `groupXX-partB-03b-ijepa-detect` | I-JEPA backbone → YOLO fine-tune @ 20% |
| 4a | `groupXX-partB-04a-dinov3-pretrain` | DINOv3 domain-adaptive pretraining |
| 4b | `groupXX-partB-04b-dinov3-detect` | DINOv3 backbone → YOLO fine-tune @ 20% |
| 5 | `groupXX-partB-05-tracking` | Tracking on YouTube/AI video |
| B | `groupXX-partB-bonus-label-efficiency` | Bonus: 10–50% label ablation |

---

## 2. Dataset & Data Partition (Leakage-Safe)

### Kaggle Paths

```
CLEANED_PATH = "/kaggle/input/datasets/sadikshahriar/agropest-12-cleaned/AgroPest-12-Cleaned"
```

### 10/10/80 Split Protocol

Merge all images → re-partition once with fixed seed:

1. **Test holdout — 10%** (~1,311 images). Never in SSL pool or fine-tuning.
2. **Validation — 10%** (~1,311 images). For model selection, early stopping. Excluded from SSL pool.
3. **SSL pool — 80%** (~10,485 images). Labels discarded. SSL pretraining only.
4. **Fine-tune subset — 20% of total** (~2,621 images). Sampled (seeded) from SSL pool. Labels restored.

> Every notebook must print partition sizes and verify (by filename intersection) that test/val are disjoint from the SSL pool.

### Class Names

```python
CLASS_NAMES = [
    "Ants", "Bees", "Beetles", "Caterpillars",
    "Earthworms", "Earwigs", "Grasshoppers", "Moths",
    "Slugs", "Snails", "Wasps", "Weevils"
]
NUM_CLASSES = 12
```

---

## 3. Best Part-A Model: YOLOv12-m

**Selected as best** among YOLOv10-m, YOLOv12-m, YOLOv26-m (highest test mAP50-95). RF-DETR excluded from Part B.

### Test Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.866 |
| Recall | 0.756 |
| mAP50 | 0.810 |
| mAP50-95 | **0.477** |
| F1 | 0.801 |

### Training Config (reuse in all Part B downstream notebooks)

```python
# Model
MODEL_NAME   = "yolo12m.pt"       # COCO-pretrained baseline
ARCHITECTURE = "yolov12m"

# Training
EPOCHS        = 50
BATCH         = 32
IMGSZ         = 640
DEVICE        = [0, 1]            # DDP on 2xT4
AMP           = True
OPTIMIZER     = "auto"            # >10k iter → MuSGD lr=0.01 (or AdamW with auto)
LR0           = 0.01
LRF           = 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR  = 0.1

# Augmentation
CLOSE_MOSAIC  = 10
MOSAIC        = 1.0
MIXUP         = 0.15
DEGREES       = 10.0
SCALE         = 0.5
SHEAR         = 0.0
TRANSLATE     = 0.1
FLIPLR        = 0.5
FLIPUD        = 0.5
HSV_H         = 0.01
HSV_S         = 0.4
HSV_V         = 0.4

# Regularisation
RECT          = False
COS_LR        = False
PATIENCE      = 25
NBS           = 64
WORKERS       = 4

# Inference
CONF          = 0.25
IOU           = 0.7
MAX_DET       = 300
```

> **Fair Comparison Rule:** All Part B downstream runs must keep these identical. Only backbone initialisation and label fraction (ρ) may vary. See §2 of the assignment PDF.

---

## 4. `ssl-detection-lab` Library Structure & Mapping

The library at `../ssl library/` (v0.8.0) by Dr. Rifat provides pre-built modules for every Part B task.

### Module → Task Mapping

| `ssl-detection-lab` Module | Part B Task | Key Functions/Classes |
|---|---|---|
| `ssldet.ssl` | SSL objectives (SimCLR, BYOL, MoCo, DINOv2/v3, MAE, I-JEPA) | `create_ssl_module()`, `SSLMethod`, `available_ssl_modules()` |
| `ssldet.backbones` | YOLO, DINOv2, DINOv3 feature encoders | YOLO backbone wrappers, ViT encoders |
| `ssldet.detection` | Ultralytics detector adapter | Detector protocol, Ultralytics wrapper |
| `ssldet.evaluation` | mAP, precision, recall, PR curves | Standard COCO eval metrics |
| `ssldet.video` | Streaming detection/tracking, video metrics | ByteTrack, BoT-SORT, MOT metrics |
| `ssldet.downstream` | SSL checkpoint → YOLO weight surgery | Backbone transfer, layer mapping |
| `ssldet.workflow` | Dry-run helpers, distributed launch | `assert_supported_runtime()`, `runtime_report()` |
| `ssldet.data` | Data loaders, augmentations, view generators | SSL view pipelines, dataloaders |
| `ssldet.trainer` | Training loop, checkpointing, logging | Pretrain trainer, epoch callbacks |

### CLI Tools

```bash
ssldet doctor              # Inspect environment
ssldet-pretrain --config   # Run SSL pretraining from YAML
ssldet models              # List available backbones
```

---

## 5. Example Notebooks → Your Notebooks Mapping

The library contains 8 tutorial notebooks at `../ssl library/output/jupyter-notebook/`. They use a **football dataset** but the code structure is a direct template for AgroPest-12.

| Tutorial Notebook | Template For | What to Change |
|---|---|---|
| `simclr_football_t4x2_tutorial.ipynb` | **1a** — SimCLR pretrain | Dataset → SSL pool (80%), classes → 12, encoder → YOLOv12 backbone |
| `byol_football_t4x2_tutorial.ipynb` | **2a** — BYOL pretrain | Same as above |
| `ijepa_yolo26_football_ssl_tutorial.ipynb` | **3a** — I-JEPA pretrain | Same + ViT adapter config |
| `dinov3_yolo26_football_ssl_tutorial.ipynb` | **4a** — DINOv3 pretrain | Same + use released DINOv3 weights, domain-adaptive |
| `simclr_yolo26_football_downstream_tutorial.ipynb` | **1b** — SimCLR detect | Dataset → 20% subset, model → YOLOv12-m, eval on test |
| `ijepa_yolo26_downstream_tutorial.ipynb` | **3b** — I-JEPA detect | Same as above |
| `dinov3_yolo26_downstream_tutorial.ipynb` | **4b** — DINOv3 detect | Same as above |
| `ssl_detection_lab_football_t4x2_tutorial.ipynb` | Reference | Library overview, API patterns |

### Key Adaptations Needed

1. **Dataset paths**: Replace football dataset with `CLEANED_PATH`
2. **Classes**: nc=12, CLASS_NAMES from AgroPest-12
3. **YOLO version**: Tutorials use YOLOv26 — change to `yolo12m.pt`
4. **Config params**: Use your YOLOv12-m config from §3 (batch=32, 640x640, etc.)
5. **SSL pool**: Build the 80% unlabelled partition with seed
6. **20% subset**: Sample from SSL pool with same seed
7. **Checkpoint export**: Save to `/kaggle/working/`, publish as Kaggle dataset

---

## 6. YAML Config Files

The library has T4-optimised YAML configs at `../ssl library/examples/`:

| File | For |
|------|-----|
| `simclr_t4x2.yaml` | SimCLR pretraining params |
| `ijepa_t4x2.yaml` | I-JEPA pretraining params |
| `dinov3_yolo26_t4x2.yaml` | DINOv3 + YOLO config |
| `dinov2_t4x2.yaml` | DINOv2 (alt) |

These can be loaded with `ssldet-pretrain --config examples/simclr_t4x2.yaml` or used as reference for manual config cells.

---

## 7. Notebook-by-Notebook Implementation Guide

### Notebook Pair 1a/1b — SimCLR

**1a (Pretrain):**
- Load SSL pool (80% of data), discard labels
- Use `ssldet.ssl.create_ssl_module("simclr", encoder=..., feature_dim=..., temperature=0.2)`
- Two-augment view pipeline (random crop, colour jitter, Gaussian blur)
- Train 30 epochs, log NT-Xent loss, plot curve
- Diagnostics: t-SNE + nearest-neighbour retrieval
- Export `simclr_backbone.pt` → publish as Kaggle dataset

**1b (Detect):**
- Attach SimCLR checkpoint dataset
- Build 20% labelled subset from SSL pool (seeded)
- Load checkpoint into YOLOv12-m backbone via `ssldet.downstream`
- Print layer mapping report (which layers from SSL, which random)
- Fine-tune 50 epochs with YOLOv12-m config
- Evaluate on 10% test holdout
- Also run: random-init baseline + COCO-pretrained baseline at 20%

### Notebook Pair 2a/2b — BYOL

Same structure as SimCLR but:
- `create_ssl_module("byol", encoder=..., feature_dim=..., hidden_dim=...)`
- Online + target network with EMA update (momentum=0.996)
- Loss: cosine similarity between online predictor and target projection
- No negative pairs

### Notebook Pair 3a/3b — I-JEPA

**Integration route choices:**
1. **ViT adapter**: Pretrain ViT encoder, attach feature-pyramid adapter mapping patch tokens → P3/P4/P5
2. **Distillation**: Distil frozen ViT features into YOLO CNN backbone

- `create_ssl_module("ijepa", encoder=..., context_size=..., target_size=...)`
- Predict representation of masked target blocks from context blocks
- No hand-crafted augmentations

### Notebook Pair 4a/4b — DINOv3

- Start from released DINOv3 pretrained weights
- Continue domain-adaptive pretraining on SSL pool
- Student-teacher self-distillation with multi-crop
- Gram anchoring for stable dense features
- ViT → YOLO integration via adapter or distillation

### Notebook 5 — Tracking

- Load best SSL backbone detector (highest test mAP50-95 from 1b–4b)
- ByteTrack or BoT-SORT via `ssldet.video` / Ultralytics `track()` interface
- Video: YouTube clip with target pest class (or AI-generated)
- Report MOTA, MOTP, IDF1 (if GT available) or proxy metrics
- Analyse failure modes: occlusion, motion blur, domain gap

### Bonus Notebook — Label-Efficiency Ablation

- Sweep ρ = 10%, 20%, 30%, 40%, 50% (nested subsets, seeded)
- Two initialisations: best SSL backbone (from Task 1 winner) vs COCO-pretrained
- 10 runs total (5 fractions × 2 init), 50 epochs each
- Headline figure: mAP50-95 vs label fraction with Part A 100% reference line
- Break-even analysis

---

## 8. Checkpoint Hand-off Pattern

```
Pretraining notebook                          Downstream notebook
──────────────────────────                    ──────────────────────────
save to /kaggle/working/                      load from /kaggle/input/
  simclr_backbone.pt      ── publish as ──>   groupXX-partb-ssl-checkpoints/
  byol_backbone.pt        Kaggle dataset        simclr_backbone.pt
  ijepa_backbone.pt                             byol_backbone.pt
  dinov3_backbone.pt                            ijepa_backbone.pt
                                                dinov3_backbone.pt
```

Naming convention: `<method>_backbone.pt` for pretrained, `<method>_detector.pt` for fine-tuned.

---

## 9. Quick Start

```python
# In Kaggle notebook, install the library
!pip install ssl-detection-lab

# Or install from the cloned repo
!pip install /kaggle/input/ssl-detection-lab

# Import
from ssldet.ssl import create_ssl_module, available_ssl_modules
from ssldet.downstream import load_backbone_into_yolo
from ssldet.evaluation import evaluate_detector
from ssldet.video import track_video
from ssldet.workflow import assert_supported_runtime, runtime_report

# Check environment
print(runtime_report())
assert_supported_runtime(require_cuda=True, minimum_gpus=2)
```

---

## 10. Viva Preparation

Every CONFIG parameter must be explainable. For each notebook, be ready to answer:
- Why this learning rate / batch size / augmentation choice?
- Why this SSL method works (or doesn't) for pest detection?
- How does the data partition prevent leakage?
- Why was this YOLO model chosen over others?
- What do the t-SNE / retrieval diagnostics reveal?

See `Lab Assignment Part B.md` §4.3 for per-method viva questions.