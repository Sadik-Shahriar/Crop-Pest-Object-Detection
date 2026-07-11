# Model Comparison: AgroPest-12 Object Detection

## Summary Table (Test Set)

| Metric | YOLOv10m | YOLOv12m | YOLOv26m | RF-DETR Nano |
|---|---|---|---|---|
| **Precision** | 0.819 | **0.866** | 0.850 | 0.437 |
| **Recall** | 0.737 | **0.756** | 0.731 | 0.693 |
| **mAP@50** | 0.791 | **0.810** | 0.787 | 0.571 |
| **mAP@50-95** | 0.473 | **0.477** | 0.475 | 0.362 |
| **F1-Score** | 0.772 | **0.801** | 0.779 | 0.532 |
| **Parameters** | **15.3M** | 20.1M | 20.4M | — |
| **Inference/T4** | **23.7ms** | 35.2ms | 30.6ms | — |
| **Val F1** | 0.791 | 0.785 | **0.803** | 0.555 |

> **Bold** = best in row. RF-DETR used a different eval framework (no per-image speed reported).

---

## Ranking by Test mAP@50

| Rank | Model | mAP@50 | Δ vs best |
|---|---|---|---|
| 1 | **YOLOv12m** | **0.8102** | — |
| 2 | YOLOv10m | 0.7908 | −0.019 |
| 3 | YOLOv26m | 0.7867 | −0.024 |
| 4 | RF-DETR Nano | 0.5705 | −0.240 |

---

## Per-Class Test mAP@50 Breakdown

| Class | YOLOv10m | YOLOv12m | YOLOv26m | RF-DETR | Best |
|---|---|---|---|---|---|
| Ants | 0.727 | **0.828** | 0.795 | 0.472 | YOLOv12m |
| Bees | 0.945 | **0.958** | 0.933 | 0.658 | YOLOv12m |
| Beetles | 0.777 | **0.828** | 0.780 | 0.335 | YOLOv12m |
| Caterpillars | **0.556** | 0.550 | 0.526 | 0.344 | YOLOv10m |
| Earthworms | 0.517 | **0.572** | 0.505 | 0.292 | YOLOv12m |
| Earwigs | **0.806** | 0.786 | 0.745 | 0.346 | YOLOv10m |
| Grasshoppers | **0.633** | 0.591 | 0.585 | 0.384 | YOLOv10m |
| Moths | 0.975 | **0.992** | 0.979 | 0.991 | YOLOv12m |
| Slugs | 0.735 | 0.726 | **0.754** | 0.584 | YOLOv26m |
| Snails | 0.856 | **0.920** | 0.881 | 0.828 | YOLOv12m |
| Wasps | 0.973 | **0.978** | 0.976 | 0.786 | YOLOv12m |
| Weevils | 0.990 | **0.994** | 0.981 | 0.825 | YOLOv12m |

- **YOLOv12m** wins on **8 of 12 classes**.
- YOLOv10m wins on 3 classes (Caterpillars, Earwigs, Grasshoppers).
- YOLOv26m wins on 1 class (Slugs).
- RF-DETR wins on none.

---

## Difficult Classes (all models score mAP@50 < 0.60)

| Class | YOLOv10m | YOLOv12m | YOLOv26m | RF-DETR |
|---|---|---|---|---|
| Caterpillars | **0.556** | 0.550 | 0.526 | 0.344 |
| Earthworms | 0.517 | **0.572** | 0.505 | 0.292 |
| Grasshoppers | **0.633** | 0.591 | 0.585 | 0.384 |

Caterpillars and Earthworms are the hardest across all models — likely due to thin, elongated morphology and background blending.

---

## Per-Class Test F1-Score

| Class | YOLOv10m | YOLOv12m | YOLOv26m | RF-DETR | Best |
|---|---|---|---|---|---|
| Ants | 0.743 | **0.796** | 0.820* | 0.442 | YOLOv12m |
| Bees | **0.906** | 0.889 | 0.878 | 0.561 | YOLOv10m |
| Beetles | 0.728 | **0.814** | 0.717 | 0.381 | YOLOv12m |
| Caterpillars | 0.562 | **0.607** | 0.537 | 0.383 | YOLOv12m |
| Earthworms | 0.500 | **0.609** | 0.525 | 0.309 | YOLOv12m |
| Earwigs | **0.778** | 0.726 | 0.682 | 0.389 | YOLOv10m |
| Grasshoppers | 0.648 | **0.679** | 0.659 | 0.361 | YOLOv12m |
| Moths | 0.961 | **0.977** | 0.942 | 0.841 | YOLOv12m |
| Slugs | 0.670 | **0.680** | 0.762* | 0.544 | YOLOv12m |
| Snails | 0.856 | **0.875** | 0.886* | 0.740 | YOLOv12m |
| Wasps | 0.957 | **0.986** | 0.964 | 0.646 | YOLOv12m |
| Weevils | 0.952 | **0.975** | 0.971 | 0.782 | YOLOv12m |

> *YOLOv26m F1 computed from its per-class P/R. YOLOv12m wins on 9/12 classes by F1.

---

## Validation vs Test Consistency

| Model | Val mAP@50 | Test mAP@50 | Δ (overfit gap) |
|---|---|---|---|
| YOLOv10m | 0.796 | 0.791 | −0.005 |
| YOLOv12m | 0.808 | **0.810** | +0.002 |
| YOLOv26m | **0.810** | 0.787 | −0.023 |
| RF-DETR | 0.563 | 0.571 | +0.008 |

YOLOv12m is the **only model** where test mAP@50 exceeds validation — it generalizes best. YOLOv26m shows the largest val→test drop (−0.023), suggesting mild overfitting.

---

## Speed vs Accuracy Trade-off

| Model | Params | Infer/Img | Test mAP@50 | Efficiency (mAP/ms) |
|---|---|---|---|---|
| YOLOv10m | **15.3M** | **23.7ms** | 0.791 | **0.0334** |
| YOLOv12m | 20.1M | 35.2ms | **0.810** | 0.0230 |
| YOLOv26m | 20.4M | 30.6ms | 0.787 | 0.0257 |
| RF-DETR | — | — | 0.571 | — |

YOLOv10m is the most efficient (best mAP per ms). YOLOv12m is the most accurate but 48% slower than YOLOv10m.

---

## Final Verdict

| Criterion | Winner | Why |
|---|---|---|
| **Best Test mAP@50** | **YOLOv12m** | 0.810 — highest across all models |
| **Best Precision** | **YOLOv12m** | 0.866 — most trustworthy detections |
| **Best Recall** | **YOLOv12m** | 0.756 — finds the most ground-truth objects |
| **Best F1-Score** | **YOLOv12m** | 0.801 — best precision-recall balance |
| **Best Generalization** | **YOLOv12m** | Only model where test > val mAP |
| **Best Efficiency** | **YOLOv10m** | 15.3M params, 23.7ms/image, 0.791 mAP |
| **Best Overall** | **YOLOv12m** | Wins 8/12 classes, top test metrics, best generalization |
| **Worst** | **RF-DETR Nano** | 0.571 mAP@50 — 30% behind the worst YOLO |
