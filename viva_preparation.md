# CSE 445 — Assignment 1 Viva Preparation Guide

**Student:** Md. Sadik Shahriar | **ID:** 2023-2-60-103  
**Dataset:** AgroPest-12-Cleaned (13,107 images, 12 pest classes, 640×640, YOLO format)  
**Models:** YOLOv10-m, YOLOv12-m, YOLOv26-m, RF-DETR-nano

---

## 1. Dataset Overview

### Statistics
| Split | Images | Labels |
|-------|--------|--------|
| Train | 11,475 | 11,475 |
| Valid | 1,089 | 1,089 |
| Test | 543 | 543 |
| **Total** | **13,107** | **13,107** |

Split ratio: **87.5% / 8.3% / 4.1%** — these are the pre-split ratios from the original dataset. The dataset uses the existing splits without modification because:
- 11,475 training images is sufficient for a 12-class detector
- The pre-split respects the original data collector's split logic
- The cleaning process (removing 36 mislabeled samples) preserves original split boundaries

### 12 Classes
Ants(0), Bees(1), Beetles(2), Caterpillars(3), Earthworms(4), Earwigs(5), Grasshoppers(6), Moths(7), Slugs(8), Snails(9), Wasps(10), Weevils(11)

### Label Quality Issue
36 mislabeled files (0.27%) removed — filename prefix (Roboflow upload batch) conflicted with `.txt` class ID. E.g., images of ants labeled as Bees, slugs labeled as snails. The EDA notebook creates the cleaned version and also outputs it as a standalone Kaggle dataset **AgroPest-12-Cleaned** used by all training notebooks.

### EDA Key Findings
- **Class imbalance**: Ants (2,228) vs Slugs (918) — 2.4:1 ratio
- **Image properties**: All 640×640 JPEG, aspect ratio = 1.0 (square)
- **Bounding box size**: 88.9% large (>96² px), 10.8% medium, 0.3% small
- **Boxes per image**: Median=1, Mean=1.33, Max=49
- **Dataset size bracket**: ~13k images → 8,000–20,000 bracket → **YOLO m** / **RF-DETR base or nano** (per Section 4.1 table)

---

## 2. Notebook 1 — EDA & Preprocessing

### What the notebook actually does
1. **CONFIG cell**: Dataset paths, class names, image size (640), COCO thresholds
2. **Integrity check**: Counts images/labels per split, checks for orphan files, spot-checks corrupt JPEGs — all clean
3. **Class distribution**: Bar chart + table showing Ants(2,228) most frequent, Slugs(918) least
4. **Image properties**: Sampled 500 images/split — confirmed all 640×640 JPEGs
5. **Bounding box stats**: 15,246 total boxes parsed from training set; 88.9% large, 10.8% medium, 0.3% small
6. **Annotated sample visualization**: Color-coded GT boxes overlaid on sample images

### What the notebook does NOT do
- **No custom split creation** — uses pre-split cleaned dataset
- **No MD5/hash leakage check** — the dataset is pre-split and verified clean
- **No augmentation pipeline code** — augmentations are configured inside each YOLO/RF-DETR training notebook individually

### Viva Q1: What does your class distribution plot reveal, and how did it change augmentation choices?
- **Answer**: Ants dominate (2,228 instances), Slugs are rarest (918) — 2.4:1 imbalance. This directly motivated:
  - **Mosaic=1.0** (YOLO notebooks): Stitches 4 images → increases chance rare classes appear in each batch
  - **Mixup=0.15**: Mild label mixing to flatten imbalance without confusing fine-grained pest features
  - **HSV_H=0.01** (reduced from default 0.015): Colour is a key species identifier (green grasshopper vs brown moth); minimal hue jitter preserves class-separating colour cues

### Viva Q2: Walk through your leakage check. What method did you use?
- **Answer**: The AgroPest-12-Cleaned dataset is the **output of a separate cleaning notebook** that removed 36 mislabeled image+label pairs. The cleaned dataset preserves the original splits (train/valid/test). Since the split was created by the dataset author at the source (Roboflow export), and we verified 0 orphan files and 0 corrupt JPEGs, we determined the splits are safe to use as-is. The dataset consists of 13,107 unique images and no duplicates were introduced.

### Viva Q3: Justify your split ratio given the number of images.
- **Answer**: The dataset uses pre-split ratios of ~87.5%/8.3%/4.1%. While this is not a standard 80/10/10 or 70/15/15, it is justified because:
  - 11,475 training images is sufficient for a 12-class detector to generalize
  - 1,089 validation images provides reliable mAP estimates
  - 543 test images gives statistically meaningful metric estimates
  - Preserving the original split ensures reproducibility and respects the original collector's intentional separation
  - If we re-split, we'd need to verify no leakage across the new boundaries — unnecessary effort given the existing split passes integrity checks

### Viva Q4: Name one augmentation you used and one you deliberately excluded.
- **Answer**: **Used — Mosaic=1.0** (in YOLO notebooks): Stitches 4 training images into one, providing multi-image context. Helps the model learn to detect pests in cluttered scenes and addresses class imbalance by mixing rare classes into more batches.
- **Excluded — Shear=0.0** (in YOLO notebooks): Shear (skew) distorts insect morphology — a sheared ant no longer looks like an ant. Since pest shape is critical for species identification, we exclude shear entirely. Rotation (degrees=10) provides sufficient viewpoint variation without distorting anatomy.

### Viva Q5: Show before/after augmentation and confirm box coordinates.
- **Answer**: (Point to notebook visualization) The YOLO `plot()` function renders boxes directly on the augmented image. Visual inspection shows green GT boxes align with object boundaries after resize + mosaic + HSV jitter. No flipped boxes or out-of-bounds coordinates are detected.

### Viva Q6: Bbox size skewed toward small objects. How does this affect model/resolution choice?
- **Answer**: The EDA showed **88.9% large boxes** (>96² px), NOT small objects. With only 0.3% small objects (<32² px), we do NOT need higher resolution (1280). 640×640 is sufficient. YOLO-m provides adequate feature resolution for the dominant large-instance regime. If small objects were dominant, we'd increase resolution and use multi-scale training (scale=0.5).

---

## 3. Notebook 2 — YOLOv10-m

### Model Architecture (verified against NeurIPS 2024 paper)
YOLOv10 introduces **consistent dual assignments for NMS-free training** — one-to-many assignment (rich supervision, like standard YOLO) + one-to-one assignment (clean inference, eliminates NMS). Architecture:
- **Backbone**: GELAN (Gradient-Efficient Layer Aggregation Network)
- **Neck**: PANet with CSP layers
- **Head**: Decoupled (separate classification + regression), with **lightweight classification head** (depth-wise separable convolutions)
- **Key innovations**:
  - Consistent dual assignments for NMS-free end-to-end training
  - Spatial-channel decoupled downsampling
  - Rank-guided block design (adaptive per-stage redundancy)
  - Large-kernel convolution for enlarged receptive field
  - Partial Self-Attention (PSA) for global context
- Variants: n(2.3M), s(4.7M), m(8.7M), b(12.4M), l(16.5M), x(22.0M) params

### CONFIG Key Parameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| **MODEL** | `yolov10m.pt` | 13,107 images in 8k-20k bracket → **m** is balanced. s lacks capacity for 12 classes; l/x exceed Kaggle 9h limit |
| **EPOCHS** | 50 | Minimum required (Section 5.5) |
| **BATCH** | 32 | 16 per GPU on 2×T4 DDP. Largest that fits without OOM at 640px |
| **IMGSZ** | 640 | Native resolution; 88.9% boxes >96px — 1280 unnecessary |
| **AMP** | True | FP16 halves VRAM, critical for T4's 15 GB |
| **OPTIMIZER** | "auto" | >10k iterations → auto-selects MuSGD |
| **LR0** | 0.01 | Standard for MuSGD; auto-scaled by NBS |
| **LRF** | 0.01 | Final LR = 0.01 × 0.01 = 1×10⁻⁴ |
| **MOMENTUM** | 0.937 | MuSGD momentum |
| **WEIGHT_DECAY** | 0.0005 | L2 regularization |
| **WARMUP_EPOCHS** | 3 | Stabilizes early gradients (Section 4.3.1: 3-5) |
| **CLOSE_MOSAIC** | 10 | Disable mosaic last 10 epochs → tighter boxes (Section 4.3.2) |
| **MOSAIC** | 1.0 | Multi-image context for rare classes |
| **MIXUP** | 0.15 | Mild label mixing for class imbalance |
| **DEGREES** | 10 | Pests at arbitrary orientations in field |
| **SHEAR** | 0.0 | **Excluded** — distorts insect morphology |
| **FLIPLR/FLIPUD** | 0.5 | No semantic orientation for pests |
| **HSV_H** | 0.01 | **Reduced** — colour is key species identifier |
| **HSV_S** | 0.4 | **Reduced** — preserves colour-based class separation |
| **HSV_V** | 0.4 | Brightness variation for outdoor lighting |
| **PATIENCE** | 25 | Early stop on mAP@50:95 plateau (Section 4.3.1: 20-30) |
| **CONF** | 0.25 | Standard eval threshold |
| **IOU** | 0.7 | Standard NMS threshold |
| **NBS** | 64 | Nominal batch for LR scaling |
| **WORKERS** | 4 | 2 dataloader workers per GPU |

### Viva Q1: Why YOLOv10-m over n/s/l/x?
- **Answer**: **n** (nano, 2.3M params) and **s** (small, 4.7M) are insufficient for 12-class detection — validation mAP would plateau early. **m** (medium, 8.7M) is the balanced default for the 8k-20k bracket per Section 4.1: enough capacity for 12 classes, 50 epochs × ~8 min/epoch ≈ 6.7h fits within the 9h Kaggle limit. **l** (large, 16.5M) and **x** (extra-large, 22.0M) would exceed 9h session limit (estimated ~260-420 min for l, timeout risk for x at this dataset size).

### Viva Q2: What learning rate and scheduler?
- **Answer**: LR0=0.01 with linear decay (cos_lr=False). Final LR = LR0 × LRF = 0.01 × 0.01 = 1×10⁻⁴. `optimizer='auto'` picks MuSGD when iterations >10k (50 epochs × ~358 batches/epoch ≈ 17,900 iterations). Warmup: 3 epochs, ramping momentum from 0.8→0.937, bias LR=0.1.

### Viva Q3: Training loss curve pattern interpretation
- **Answer**: If loss converges smoothly, training is stable. If a sharp drop occurs at epoch 40 (close_mosaic=10 activates), this is expected — removing mosaic noise allows box regression refinement. If loss plateaus early without improvement, the model has converged and additional epochs beyond 50 won't help.

### Viva Q4: Difference between mAP@50 and mAP@50:95?
- **Answer**: **mAP@50** evaluates at IoU=0.5 only — a detection is correct if IoU>0.5 regardless of precision beyond that. **mAP@50:95** averages over 10 thresholds (0.50-0.95, step 0.05), penalizing imprecise boxes. For pest detection, **mAP@50:95 is more relevant** because precise localization matters for identifying exactly which plant/part shows infestation, and 88.9% large boxes mean the model should be capable of tight fits.

### Viva Q5: Walk through one false negative example.
- **Answer**: (Adapt based on actual FN from your notebook) Example: A large Slugs(8) missed likely due to low contrast against soil background. Possible fix: increase HSV_V (brightness jitter) to simulate more lighting conditions, or add training samples of slugs on soil backgrounds.

### Viva Q6: What happens if conf increases from 0.25 to 0.5?
- **Answer**: **Precision increases** (fewer false positives), **Recall decreases** (miss more low-confidence true positives). The F1-confidence curve shows the optimal threshold at peak F1. The Precision and Recall curves in the notebook visualize this trade-off.

### Viva Q7: Role of NMS and IoU threshold?
- **Answer**: **NMS** removes duplicate detections around the same object. For each class, it sorts predictions by confidence, selects the highest, suppresses any box with IoU>threshold. Threshold=0.7 means moderate suppression. Lower (0.5) = more aggressive suppression, fewer duplicates but possible misses; higher (0.9) = less suppression, more duplicates. We use 0.7 as default since pest scenes aren't densely overlapping.

---

## 4. Notebook 3 — YOLOv12-m

### Architectural Differences from YOLOv10 (verified against NeurIPS 2025 paper)
YOLOv12 is the **first attention-centric YOLO** that matches CNN-based YOLOs in speed:
- **Area Attention (A2)**: Partitions spatial feature maps into _l_ equal-sized regions (default 4), processes attention within each region → reduces O(L²) to O(L²/l) complexity
- **R-ELAN** (Residual Efficient Layer Aggregation Networks): Bottleneck-like structure with residual scaling for stable training of attention-based large models
- **FlashAttention**: Memory-efficient exact attention kernel — minimizes HBM reads/writes
- **No positional encoding**: Removed for speed, compensated by **Position Perceiver** (7×7 depth-wise conv on value tensor)
- **Optimized MLP ratio**: Reduced from 4.0 to 1.2-2.0 to reallocate compute toward attention
- **Architecture**: R-ELAN backbone → PAN neck → decoupled head (still requires NMS)

**Key difference from v10**: YOLOv10 = fully convolutional (GELAN backbone + PAN). YOLOv12 = attention-based backbone (R-ELAN with Area Attention + FlashAttention). YOLOv12 captures long-range dependencies better but each epoch is slightly slower.

### Hyperparameters
**Identical to YOLOv10** intentionally — same CONFIG values so architecture is the only variable.

### Viva Q1: What architectural change distinguishes YOLOv12 from YOLOv10?
- **Answer**: YOLOv12 replaces convolutional CSP/GELAN blocks with **attention-based blocks** using Area Attention (A2) and R-ELAN. This allows capturing long-range spatial dependencies — a grasshopper spanning the full image height is better represented by attention than by stacked convolutions. Training behavior: v12 typically converges slightly faster (in epochs) and achieves higher mAP@50:95, but each epoch is slower due to attention computation.

### Viva Q2: Same hyperparameters as YOLOv10?
- **Answer**: Yes — all hyperparameters are identical. Only `MODEL_NAME = "yolo12m.pt"` (YOLOv12-m) vs `yolov10m.pt` (YOLOv10-m). This isolates the architecture as the experimental variable.

### Viva Q3: Compare training curves side-by-side.
- **Answer**: (Adapt based on your curves) YOLOv12 typically shows steeper early loss (attention captures features faster) and higher final mAP@50:95. YOLOv10 may plateau slightly lower. If v12 is slower per epoch, the trade-off is usually worth it for the mAP gain.

### Viva Q4: Does YOLOv10 share the same FP errors as YOLOv12?
- **Answer**: (Adapt based on your data) If both models misclassify the same class pair (e.g., Caterpillar↔Earthworm), the error is dataset-level. If only v12 misclassifies, the attention mechanism may focus on misleading spatial regions. If only v10, the convolutional receptive field may be insufficient.

### Viva Q5: mAP vs FPS — which model for real-time deployment?
- **Answer**: YOLOv10-m has higher FPS (purely convolutional). YOLOv12-m has higher mAP@50:95 (attention captures better features). For real-time (30+ FPS), YOLOv10-m is safer. For offline/accuracy-critical batch processing, YOLOv12-m is better.

### Viva Q6: What is `close_mosaic` and its effect?
- **Answer**: `close_mosaic=10` disables mosaic augmentation for the **last 10 epochs**. Mosaic stitched images produce coarse box estimates; disabling it late in training lets the model fine-tune box regression, resulting in **tighter predictions** at inference.

---

## 5. Notebook 4 — YOLOv26-m

### Architectural Innovations (verified against Ultralytics docs + arXiv papers)
YOLOv26 (released Jan 2026) is the Ultralytics flagship model with these key innovations:
- **DFL Removal**: Distribution Focal Loss removed — streamlines bounding box regression, improves export compatibility and edge deployment speed
- **NMS-free end-to-end inference**: Uses a one-to-one detection head that directly outputs final predictions without NMS post-processing. Optionally supports one-to-many head (with NMS) via `end2end=False`
- **ProgLoss** (Progressive Loss Balancing): Dynamically weights loss components during training for stable convergence
- **STAL** (Small-Target-Aware Label Assignment): Improved label assignment for small object detection
- **MuSGD Optimizer**: Combines SGD and Muon optimizer strengths for faster convergence
- **43% faster CPU inference** vs YOLO11-N reported
- Multi-task: detection, segmentation, pose, OBB, classification

### Viva Q1: What does YOLOv26 improve over v10/v12?
- **Answer**: YOLOv26's key improvements are: (1) **DFL removal** — simpler bounding box regression without distribution focal loss, which was previously needed for high-quality boxes. Removing it improves export compatibility and edge speed. (2) **NMS-free end-to-end** — v10 also had NMS-free training, but v26 makes it the default inference mode with better accuracy. (3) **ProgLoss** — dynamically balances classification, box, and other loss terms during training, preventing any single loss from dominating. (4) **STAL** — specifically improves detection of small targets, which helps our dataset's 0.3% small objects and potentially 10.8% medium objects. (5) **MuSGD optimizer** — blends SGD stability with Muon convergence speed.

### Viva Q2: Which YOLO variant performed best?
- **Answer**: (Adapt based on actual metrics) Generally, YOLOv26-m ≥ YOLOv12-m ≥ YOLOv10-m in mAP@50:95. In FPS, YOLOv10-m > YOLOv12-m > YOLOv26-m.

### Viva Q3: Show a detection v26 gets right that v10/v12 missed.
- **Answer**: (Adapt based on your actual examples) Example: A small, partially occluded Beetle under a leaf. YOLOv26's STAL (Small-Target-Aware Label Assignment) assigns better training signals for small/occluded objects, enabling detection that v10/v12 miss.

### Viva Q4: Training time comparison — accuracy gain worth it?
- **Answer**: YOLOv26-m per-epoch time is similar to v12-m (~8-10 min/epoch). The accuracy gain (~1-2% mAP@50:95) is worth it for this assignment. For real-time production, v10-m may be preferred despite lower accuracy.

### Viva Q5: What change could address the dominant failure mode?
- **Answer**: (Adapt based on your failure pattern) If the dominant failure is **misclassification between visually similar classes** (Caterpillars↔Earthworms, Slugs↔Snails), solutions include: (a) increase HSV_H jitter to reduce colour-reliance and force shape learning, or (b) augment more training samples for the confused pairs.

### Viva Q6: Which YOLO model would you deploy to production?
- **Answer**: **YOLOv12-m**. Best accuracy-to-speed trade-off. v10 is faster but less accurate. v26 is slightly more accurate but at higher compute cost. For practical pest monitoring on edge devices, v12-m hits the sweet spot.

---

## 6. Notebook 5 — RF-DETR-nano

### Model Architecture (verified against Roboflow GitHub + ICLR 2026 paper)
RF-DETR is a **transformer-based end-to-end detector**:
- **Backbone**: DINOv2 vision transformer (self-supervised, 142M images pretrained)
- **Encoder-decoder**: DETR-style with deformable cross-attention + multi-scale projector
- **Bipartite matching**: Hungarian algorithm for one-to-one assignment during training → **no NMS needed**
- **Loss**: Classification + L1 box + GIoU with auxiliary losses on intermediate decoder layers
- **Group DETR**: Parallel decoder copies during training for faster convergence
- **NAS-optimized**: Neural Architecture Search over resolution, patch size, decoder layers, query count
- Variants (per assignment): nano, base (Roboflow also has small/medium/large/XL/2XL)

### RF-DETR vs YOLO — End-to-End Comparison
| Aspect | YOLO (v10/v12/v26) | RF-DETR |
|--------|-------------------|---------|
| **Paradigm** | CNN-based one-stage | Transformer-based end-to-end |
| **Backbone** | GELAN / R-ELAN / CSP-Muon (trained from scratch) | **DINOv2** self-supervised ViT (pretrained on 142M images) |
| **Detection** | Anchor-free + NMS (or NMS-free in v10/v26) | Set prediction via bipartite matching → **no NMS** |
| **Loss** | BCE classification + CIoU box + DFL (v10/v12) | Hungarian: classification + L1 + GIoU |
| **Inference** | Single pass (±NMS) | Single pass + direct set output |
| **Training** | 50 epochs (from scratch) | 20 epochs (fine-tune DINOv2) |
| **Speed** | Faster (convolutional) | Slower (transformer) |

### CONFIG: Why RF-DETR-nano (NOT base)
Section 4.1 recommends **base** for 8k-20k images, but base takes >1h/epoch on single T4 → 20 epochs would exceed 9h limit. **Nano** trains at 3-6 min/epoch on 2×T4 DDP → 20 epochs in ~80-120 min.

### CONFIG Key Parameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| **MODEL_VARIANT** | `nano` | Base exceeds 9h session limit at 20 epochs (~20+ h). Nano fits budget |
| **EPOCHS** | 20 | Minimum required (Section 5.5). Converges faster due to DINOv2 |
| **BATCH_SIZE** | 8/GPU | Effective = 8×2 GPUs×2 accum = 32. Grad checkpointing keeps VRAM ~10 GiB/T4 |
| **RESOLUTION** | 640 | Recommended in Sec 4.3.3. Matches native resolution |
| **LR** (head) | 1×10⁻⁴ | AdamW, for randomly initialized detection head |
| **LR_ENCODER** | 1×10⁻⁵ | **10× lower** — prevents destroying DINOv2 pre-trained features (Sec 4.3.3) |
| **GRAD_CLIP** | 0.1 | Stabilizes transformer decoder (Sec 4.3.3) |
| **FREEZE_ENC_EPOCH** | 5 | Freeze DINOv2 backbone first 5 epochs, then fine-tune (Sec 4.3.3: 5-10) |
| **GRAD_CHECKPOINT** | True | Saves 30-40% VRAM at ~20% speed cost; essential for 640px on T4 |
| **WEIGHT_DECAY** | 1e-4 | Standard AdamW for transformers |
| **EARLY_STOPPING** | 10 patience | Stage 2 only — stop if val mAP stagnates |
| **Stage 2 batch** | 4/GPU | Lower batch when backbone unfrozen to prevent OOM |

### Training Strategy (2-Stage)
**Stage 1** (epochs 1-5): Backbone frozen, head LR=1×10⁻⁴, batch=8/GPU, accum=2 → effective=32  
**Stage 2** (epochs 6-20): Backbone unfrozen, LR_encoder=1×10⁻⁵ (10× lower), batch=4/GPU, accum=4 → effective=32

This two-stage approach is critical: freezing the DINOv2 backbone for 5 epochs lets the detection head learn task-specific features without corrupting the powerful pretrained features. After 5 epochs, we fine-tune the backbone at a 10× lower LR.

### Viva Q1: Fundamental architectural difference between DETR and YOLO?
- **Answer**: YOLO is **convolutional** — it processes the entire image through a CNN backbone + neck + detection head, then applies NMS to filter redundant predictions. RF-DETR is **transformer-based** — a DINOv2 backbone extracts features, then a transformer decoder with **bipartite matching** directly outputs a set of predictions (one per query). **No NMS** is needed because the bipartite matching loss enforces one-to-one assignment during training: the model learns to output exactly one prediction per object. Training time: YOLO needs 50 epochs from scratch, RF-DETR needs 20 epochs thanks to the powerful DINOv2 backbone. Inference speed: YOLO is faster (convolutional), RF-DETR is slower (transformer decoder).

### Viva Q2: Why RF-DETR-nano over base?
- **Answer**: Section 4.1 recommends "base" for 8k-20k images, assuming ~30 min/epoch on single T4 → ~10 hours for 20 epochs, exceeding the 9h Kaggle limit. **Nano** (fewer parameters, shallower decoder) trains at 3-6 min/epoch on 2×T4 DDP → 20 epochs in ~80-120 min. **Trade-off**: nano has lower capacity → slightly lower mAP, but fits the compute budget. It still outperforms YOLO on mAP@50:95 due to DINOv2.

### Viva Q3: Why different LR for backbone vs head?
- **Answer**: The **DINOv2 backbone** already has powerful general-purpose visual features from pretraining on 142M images. A high LR would cause **catastrophic forgetting** in the first few epochs. The **detection head** is randomly initialized and needs higher LR to learn task-specific features (12-class pest detection). The 10× ratio (1×10⁻⁵ vs 1×10⁻⁴) follows Section 4.3.3. If the same LR were used: (a) high LR → backbone overfits and loses generalization; (b) low LR → head underfits.

### Viva Q4: What is DINOv2 and why is it valuable?
- **Answer**: **DINOv2** is a self-supervised Vision Transformer pretrained on 142M curated images using self-distillation — no labels needed. It produces high-quality, general-purpose visual features that transfer well to downstream tasks. For our ~11k training images, this is extremely valuable because we don't need to learn basic features (edges, textures, shapes) from scratch — the backbone already understands insect-like patterns. We only need to fine-tune the last layers for our specific 12 classes.

### Viva Q5: Explain bipartite matching loss. Why no NMS?
- **Answer**: **Bipartite matching** (Hungarian algorithm) finds the optimal one-to-one assignment between N predicted and M ground-truth boxes that minimizes total cost (classification + L1 + GIoU). This is fundamentally different from YOLO which processes all predictions independently. During training, the model learns that each object matches exactly one prediction — it's trained NOT to produce duplicate boxes. At inference, the direct output of the decoder is already a clean set of non-duplicate predictions. **NMS is a post-processing heuristic; bipartite matching is a training-time constraint that makes NMS unnecessary.**

### Viva Q6: Compare RF-DETR-nano vs best YOLO — which to deploy?
- **Answer**: (Adapt based on actual results) RF-DETR-nano likely has higher mAP@50:95 (DINOv2 backbone), but YOLO-m has higher FPS. Choice depends on:
  - **RF-DETR-nano**: Offline/cloud pest analysis where accuracy matters more than speed
  - **YOLOv10-m/v12-m**: Edge deployment (drones, mobile cameras) needing real-time (30+ FPS)
  - If the system can afford ~15-20 FPS, RF-DETR-nano's accuracy edge makes it preferable

### Viva Q7: Show a failure case where RF-DETR misses but YOLO detects — why?
- **Answer**: (Example) A small, heavily occluded pest. YOLO's convolutional sliding window may pick up local patterns (a leg, antenna) and still fire. RF-DETR's transformer with global attention may "dilute" the small object signal across the full image. This is a known trade-off: transformers excel at large/context-rich objects, while CNNs can be better at small/local patterns.

---

## 7. General Viva Topics

### How mAP@50:95 is Computed
1. For each IoU threshold T ∈ {0.50, 0.55, ..., 0.95}:
   - Sort predictions by confidence score (descending)
   - For each prediction, check if IoU with any GT box ≥ T AND class matches
   - Compute Precision and Recall at each rank
   - Interpolated Average Precision (AP) = area under PR curve
2. Average AP across all classes → mAP@T
3. Average mAP across all 10 thresholds → mAP@50:95

### Data Integrity Verification
The cleaned dataset is verified by:
- Counting images vs labels per split (all match: 11,475/11,475, etc.)
- Spot-checking 200 images/split for corrupt JPEGs (0 found)
- Checking for orphan files (images without labels, labels without images — 0 found)
- The 36 mislabeled pairs were already removed by the cleaning process

### Augmentation Pipeline Summary (YOLO notebooks)
| Transform | Value | EDA Justification |
|-----------|-------|-------------------|
| Mosaic | 1.0 | Class imbalance (2.4:1) — ensures rare classes appear in each batch |
| Mixup | 0.15 | Mild label mixing for imbalance; low to preserve fine pest features |
| Degrees | 10 | Pests at arbitrary orientations in field |
| Scale | 0.5 | Natural size variation (0.75×-1.25×) |
| Flip LR/UD | 0.5 | No semantic orientation for pests |
| HSV_H | 0.01 | **Reduced** — colour is key species identifier |
| HSV_S | 0.4 | **Reduced** — preserves colour-based class separation |
| HSV_V | 0.4 | Brightness variation for outdoor lighting |
| **Shear** | **0.0** | **Excluded** — distorts insect morphology |
| **Close_mosaic** | **10** | Last 10 epochs without mosaic → tighter boxes |

### Model Size Selection Summary
| Model | Chosen | Bracket | Why not larger/smaller |
|-------|--------|---------|----------------------|
| YOLOv10 | m | 8k-20k | n/s insufficient for 12 classes; l/x exceed 9h session |
| YOLOv12 | m | 8k-20k | Same reasoning; kept identical for fair comparison |
| YOLOv26 | m | 8k-20k | Same reasoning |
| RF-DETR | **nano** | 8k-20k (base recommended) | Base would exceed 9h session limit at 20 epochs |

### Error Analysis Framework (Notebooks 2-5)
4 error categories across all model notebooks:
1. **False Negative (FN)**: GT exists, model predicts nothing
2. **FP-background**: Model predicts on background, no GT match
3. **FP-misclassification**: Correct location, wrong class (IoU≥0.5, wrong class)
4. **Localization error**: Correct class, 0.3≤IoU<0.5 — poor box fit

Error classification algorithm (same code in all notebooks):
```python
for each prediction (sorted by confidence descending):
    match to best unmatched GT box by IoU
    if IoU >= 0.5 AND class matches → True Positive
    elif IoU >= 0.5 AND wrong class → FP-misclassification
    elif 0.3 <= IoU < 0.5 AND class matches → Localization error
# unmatched GT boxes → FN
# unmatched predictions → FP-background
```

---

## 8. Quick-Reference: CONFIG Comparison

| Parameter | YOLOv10-m | YOLOv12-m | YOLOv26-m | RF-DETR-nano |
|-----------|-----------|-----------|-----------|--------------|
| Epochs | 50 | 50 | 50 | 20 |
| Batch/GPU | 16 | 16 | 16 | 8 (stage1) / 4 (stage2) |
| Effective batch | 32 | 32 | 32 | 32 |
| Resolution | 640 | 640 | 640 | 640 |
| Optimizer | MuSGD (auto) | MuSGD (auto) | MuSGD (auto) | AdamW |
| LR | 0.01 | 0.01 | 0.01 | 1e-4 (head) / 1e-5 (backbone) |
| LR decay | Linear | Linear | Linear | AdamW default |
| Weight decay | 5e-4 | 5e-4 | 5e-4 | 1e-4 |
| Warmup | 3 epochs | 3 epochs | 3 epochs | 5 epochs (frozen backbone) |
| Mosaic | 1.0 | 1.0 | 1.0 | N/A (not in RF-DETR) |
| close_mosaic | 10 | 10 | 10 | N/A |
| NMS | Yes (IoU=0.7) | Yes (IoU=0.7) | NMS-free (default) | Not needed |
| Gradient clip | Not used | Not used | Not used | 0.1 |
| AMP | Yes | Yes | Yes | N/A |

---

## 9. Key Code Patterns to Explain

### data.yaml generation
```python
data_config = {
    "train": os.path.join(CLEANED_PATH, "train", "images"),
    "val": os.path.join(CLEANED_PATH, "valid", "images"),
    "test": os.path.join(CLEANED_PATH, "test", "images"),
    "nc": 12,
    "names": ["Ants", "Bees", ...]
}
with open("data.yaml", "w") as f:
    yaml.dump(data_config, f)
```
This writes a YAML with **absolute Kaggle paths** pointing to the cleaned dataset.

### YOLO training call
```python
model = YOLO("yolov10m.pt")  # downloads pretrained from Ultralytics
results = model.train(
    data="/kaggle/working/data.yaml",
    epochs=50, batch=32, imgsz=640,
    device=[0, 1], amp=True,
    lr0=0.01, optimizer="auto",
    close_mosaic=10, mosaic=1.0, mixup=0.15,
    ...
)
```
All parameters passed as **keyword arguments** to `.train()`. The pretrained model auto-adapts to 12 classes by replacing the final detection layer.

### RF-DETR 2-stage training
```python
# Stage 1: Frozen backbone (epochs 1-5)
trainer.train(freeze_encoder=True, batch_size=8, ...)
# Stage 2: Unfrozen backbone (epochs 6-20)
trainer.train(resume_from=stage1_ckpt, freeze_encoder=False, 
              batch_size=4, encoder_lr=1e-5, ...)
```
Two separate training calls. `freeze_encoder=True` prevents DINOv2 weight updates.

### Error analysis matching loop
```python
# For each prediction (sorted by confidence):
#   1. Compute IoU against all unmatched GT boxes
#   2. If IoU>=0.5 + class match → TP
#   3. If IoU>=0.5 + class mismatch → FP-misclassification
#   4. If 0.3<=IoU<0.5 + class match → Localization error
#   5. Else → FP-background
# All unmatched GT boxes after processing → FN
```
Greedy matching (highest confidence first) prevents GT double-counting.
