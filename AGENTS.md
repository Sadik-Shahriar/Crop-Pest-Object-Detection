# CSE 445 — Assignment 1: Object Detection

## Project

Train & evaluate 4 detectors (YOLOv10, YOLOv12, YOLOv26, RF-DETR) on AgroPest-12. Submit 5 public Kaggle notebooks by **July 12, 2026**. Assessment is viva-based — every CONFIG parameter must be explainable.

## Repo Layout

```
code/          → notebooks go here (copy to Kaggle)
context/       → dataset context + assignment PDF (read before starting)
reference/     → prior YOLO notebook for coding patterns
AgroPest-12 Image Dataset for Crop Pest Detection/  → dataset (gitignored, 537 MB)
```

## Dataset

- **12 classes**: Ants(0), Bees(1), Beetles(2), Caterpillars(3), Earthworms(4), Earwigs(5), Grasshoppers(6), Moths(7), Slugs(8), Snails(9), Wasps(10), Weevils(11)
- **13,143 images** (11,502 train / 1,095 valid / 546 test), 640x640, YOLO format
- Kaggle raw path: `/kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset/`
- **Clean dataset (use for training)**: `AgroPest-12-Cleaned` — Kaggle path: `/kaggle/input/datasets/sadikshahriar/agropest-12-cleaned/AgroPest-12-Cleaned` (13,107 images)
- Local path: `AgroPest-12 Image Dataset for Crop Pest Detection/`
- Full context: `context/AgroPest-12_Dataset_Context.md`
- I will make the notebooks for kaggle which will have 2x T4 gpus

### Dataset Quality Note

**36 files (0.27%) have confirmed label errors** — filename prefix implies one class but the `.txt` label says another (e.g., ants images labeled as Bees, bee images labeled as Beetles). These are Roboflow export errors, not annotation ambiguity.

- Affected batches: `ants-1/43/44/442`, `bees-103/10`, `slug-101/109/127`, `catterpillar-297/298/353` + 3 others
- After cleaning: 13,107 clean images (27 train + 6 valid + 3 test removed)
- The EDA notebook (Notebook 1) creates `/kaggle/working/AgroPest-12-Cleaned/` and removes the 36 bad pairs
- That output is saved as a standalone Kaggle dataset named **AgroPest-12-Cleaned**
- All training notebooks (2–5) add **AgroPest-12-Cleaned** as an input dataset and use it directly



## Agent Workflow

### Model Assignment
- **Main chat** → `deepseek/deepseek-v4-pro` (paid, higher quality)
- **Sub-agents** (explore, general, scout) → `opencode/deepseek-v4-flash-free` (free)

### Task Split for Token Efficiency

| Task | Agent | Why |
|------|-------|-----|
| Markdown cells, documentation, cell descriptions | Sub-agent (free) | Low-risk text, no impact on accuracy |
| File searches, data exploration, counting | Sub-agent (free) | Read-only, cheap, fast |
| Boilerplate code (imports, setup, utilities) | Sub-agent (free) | Simple patterns, easy to verify |
| Research (ultralytics API, RF-DETR docs, Kaggle structure) | Sub-agent (free) | Reference lookups don't need Pro |
| Complex code cells (training configs, loss curves, metrics) | Main agent (Pro) | Viva-critical quality needed |
| CONFIG cells | Main agent (Pro) | Every parameter must be explainable |
| Error analysis, failure-pattern summaries | Main agent (Pro) | Requires deep reasoning |
| Final notebook review before Kaggle upload | Main agent (Pro) | Catch subtle mistakes |

### Workflow Rules
- **Read context once per session**: Read `context/AgroPest-12_Dataset_Context.md` and `context/Lab Assignment-A.md` at session start — content stays in the window; not re-read on every prompt.
- **Sub-agents for markdown**: Use the free sub-agent to write all markdown cells (descriptions, section headers, explanations around code). This saves Pro tokens on safe work.
- **Sub-agents for research/lookups**: Before writing any config cell, use a sub-agent to research the model API (ultralytics docs, RF-DETR params). Only bring findings to the main chat.
- **Main agent for critical code**: Training loops, eval pipelines, CONFIG params, error analysis cells — always use Pro. If a sub-agent writes a complex cell, the main agent should review it.
- **Be concise**: Bullet-point answers. Skip explanations unless asked. Save tokens.
- **Check references**: Look at `reference/license-plate-detection-yolov12-xai.ipynb` for YOLO notebook structure patterns.

### MANDATORY: Use AgroPest-12-Cleaned for All Training Notebooks

**Every model training notebook (Notebooks 2–5) MUST use the `AgroPest-12-Cleaned` dataset, NOT the raw `crop-pests-dataset` input.**

## Assignment B: Shared Dataset Input

All Assignment B SSL pretraining and downstream notebooks use the published
`AgroPest-12-PartB-Safe` Kaggle dataset, attached at:

```text
/kaggle/input/datasets/sadikshahriar/agropest-12-partb-safe/AgroPest-12-PartB-Safe/
```

Example labelled fine-tuning image path:

```text
/kaggle/input/datasets/sadikshahriar/agropest-12-partb-safe/AgroPest-12-PartB-Safe/finetune_20/images/WASP-1-_jpg.rf.32a7603c15714c1930c9757db75f7643.jpg
```

### Part B Partition Protocol

- The original cleaned validation and test splits are preserved unchanged.
- The 11,475-image cleaned training split is divided into a fixed-seed (`445`),
  disjoint 80/20 split at the augmentation-family level.
- Each alphabetically consecutive triplet is one source image plus two augmented
  variants. A complete triplet belongs to either SSL or labelled fine-tuning, never both.
- The split is stratified by the triplet's dominant annotated class to preserve the
  original training class distribution.

| Partition | Kaggle-relative path | Images | Use |
|---|---|---:|---|
| SSL pool | `ssl_pool/images/` | 9,180 | Pretraining only; labels are absent |
| Fine-tuning | `finetune_20/images/`, `finetune_20/labels/` | 2,295 | All downstream model training |
| Validation | `valid/images/`, `valid/labels/` | 1,089 | Early stopping and model selection only |
| Test | `test/images/`, `test/labels/` | 543 | Final evaluation only |

### Mandatory Part B Rules

- Pretraining notebooks (`01a`–`04a`) must read only `ssl_pool/images/` and must
  never read labels, validation images, or test images.
- Downstream notebooks (`01b`–`04b`) must use the same `finetune_20`, validation,
  and test paths above for every SSL method and baseline.
- Generate a working `data.yaml` with absolute paths to the attached input; do not
  write to `/kaggle/input/`.
- Keep and cite `partition_manifest.csv`, `class_distribution_report.csv`, and
  `partition_report.json` for reproducibility and report documentation.

## Assignment B: Traditional Standalone SSL Implementation

The professor approved a traditional, self-contained PyTorch implementation for
Part B. Do **not** install, import, or depend on `ssl library/`,
`ssl-detection-lab`, `ssldet`, Lightly, or PyTorch Lightning. Keep the course
mechanisms and experimental protocol, but implement the data pipeline, SSL
objective, training loop, checkpoint handoff, and diagnostics directly.

### Installation and Runtime

```python
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "ultralytics>=8.4.96", "scikit-learn", "seaborn",
])
```

- Keep Kaggle's CUDA-matched PyTorch installation; do not install a standalone
  PyTorch wheel.
- Require Python 3.10+, PyTorch 2.4+, torchvision 0.19+, Ultralytics 8.4.96+,
  CUDA, and two visible T4 GPUs before launching DDP.
- Use `python -m torch.distributed.run --standalone --nproc_per_node=2` for
  dual-GPU training. The batch-size CONFIG value is **per GPU**.
- Use NCCL, `DistributedSampler`, `DistributedDataParallel`, AMP, finite-loss
  checks, gradient clipping, and rank-zero-only checkpoint/history writes.
- Write all generated scripts, checkpoints, plots, and reports to
  `/kaggle/working/`, never `/kaggle/input/`.

### Traditional Implementation Requirements

- Keep every parameter in one fully commented CONFIG cell. Every value must be
  explainable during viva.
- Wrap the actual YAML-defined Ultralytics backbone. Preserve YOLO graph routing
  through each layer's `f`, `i`, and model `save` attributes; do not approximate
  the backbone with a fixed `nn.Sequential` layer slice.
- Build image-only datasets directly from `ssl_pool/images`; never construct or
  inspect label, validation, or test paths in a pretraining notebook.
- Build two independent SimCLR views with TorchVision transforms and show an
  augmentation preview before training.
- Implement NT-Xent directly: concatenate the two normalized view embeddings,
  mask self-similarities, and use each view's paired augmentation as its target.
- State whether negatives are local or global. The standard standalone SimCLR
  implementation uses local negatives: with source batch `B` per GPU, each anchor
  has `2B - 2` local negatives. Gradient accumulation does not add simultaneous
  negatives.
- Use a projection MLP only for the SSL objective. Transfer only the encoder;
  never transfer the projector, optimizer, scheduler, detector neck, or detector
  head.
- Save `best_ssl.pt` for the lowest epoch SSL loss, `last_ssl.pt` for resuming,
  `history.csv`, `run_manifest.json`, a backbone-only checkpoint, and a
  detector-compatible checkpoint plus JSON transfer report. Enforce 100% backbone
  key/shape coverage during export rather than silently loading partial weights.
- Publish the complete output directory as a Kaggle dataset so the paired `*b`
  notebook can attach the exact checkpoint, config, manifest, and transfer report.

### Model and Method Rules

- Use **YOLOv12m** for every Part B downstream detector. It is the Part A winner
  by the required test mAP50-95 criterion (`0.4773`).
- For strict label-free SimCLR, BYOL, and I-JEPA student pretraining use
  `yolo12m.yaml`, not `yolo12m.pt`; this starts from random weights.
- `yolo12m.pt` is reserved for the required COCO-pretrained downstream baseline.
- All SSL notebooks use 30 epochs. All downstream and baseline fine-tuning runs
  use the same 50-epoch YOLOv12m protocol.
- Pretraining image resolution, batch size, and objective parameters may be
  method-specific and must be justified. Downstream resolution, batch size,
  optimizer/schedule, augmentations, seed, and evaluation settings must remain
  identical across methods.
- SimCLR's in-batch negatives are local to each DDP process in the standalone
  implementation;
  gradient accumulation does not add simultaneous negatives. State the per-GPU
  batch size and local contrastive batch in the CONFIG explanation.

### Implementation References

- `phase-1-ssl-224-simclr.ipynb` for the simple SimCLR notebook flow and
  augmentation-preview idea. Do not copy its dependency workaround, COCO
  initialization, or fixed sequential YOLO slice.
- `cse445-simclr-yolo26-football-player-detection.ipynb` for a direct encoder,
  projector, NT-Xent, curve, and checkpoint pattern. Adapt the model and dataset
  paths to YOLOv12m and AgroPest-12 Part B.
- The Part B assignment, partition manifest, class-distribution report, and
  partition report remain the source of truth for dataset use and fairness.

### Pretraining Notebook Checklist

1. CONFIG cell with paths, method, YOLO architecture, seed, 30 epochs, image
   size, per-GPU batch, optimizer/LR schedule, AMP, and objective parameters.
2. Runtime check and assertion that only `ssl_pool/images/` is used; never inspect
   or read labels, `valid`, or `test`.
3. One image-only augmentation/mask preview and a finite-loss/gradient sanity check.
4. DDP pretraining, `history.csv` loss plot, run manifest, and checkpoint export.
5. Image-only representation diagnostics from a fixed SSL-pool sample: t-SNE/UMAP
   plus query-and-top-five cosine nearest-neighbour panel. Do not make label-based
   claims from this diagnostic.
6. Publish the full output directory and record the exact Kaggle checkpoint path
   for the paired downstream notebook.
