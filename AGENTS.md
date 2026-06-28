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

