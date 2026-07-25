# BYOL and I-JEPA Collapse Report

## Purpose

This report documents anomalous BYOL and I-JEPA pretraining runs using
`ssl-detection-lab` 0.8.0. It is intended for the library authors/course
maintainers. The report separates confirmed implementation defects from
observed behavior that needs an author-level design decision.

## Executive Summary

Two 30-epoch, dual-T4 runs on the 9,180-image AgroPest-12 SSL pool produced
near-zero non-contrastive losses almost immediately:

| Method | Epoch 1 loss | Minimum loss | Minimum-loss epoch | Final loss | Exported backbone feature std |
|---|---:|---:|---:|---:|---:|
| BYOL | 0.68028 | 0.000112 | 3 | 0.02207 | 0.018106 |
| I-JEPA | 0.29092 | 0.002087 | 2 | 0.01335 | 0.000059 |

The I-JEPA representation standard deviation is measured across 500 distinct
images after L2 normalization. `0.000059` is effectively constant-image
representation collapse, not useful semantic organization.

The source audit found:

1. A confirmed I-JEPA positional-encoding defect: target mask tokens lose
   location identity before entering the transformer predictor.
2. A confirmed AMP trainer defect: EMA, scheduler, and global-step state
   advance even when `GradScaler` skips the optimizer update due to overflow.
3. A checkpoint-selection weakness: lowest SSL loss is always selected even
   when the lowest loss is caused by representation collapse.
4. A BYOL collapse-risk design deviation: the implementation uses LayerNorm
   heads with no variance/covariance constraint or collapse detector. The run
   strongly suggests projector-level degeneration, but this is not claimed to
   be a single-line proven defect without an ablation after the fixes above.

## Reproduction Environment

| Item | Value |
|---|---|
| Library | `ssl-detection-lab 0.8.0` |
| PyTorch | `2.10.0+cu128` |
| Ultralytics | `8.4.105` |
| Hardware | Kaggle Tesla T4 x2 |
| Dataset | AgroPest-12-PartB-Safe `ssl_pool/images` |
| Images | 9,180, labels absent from the SSL pool |
| Student architecture | `yolo12m.yaml`, random initialization |
| Resolution | 224 x 224 |
| Epochs | 30 |
| Seed | 445 |
| Optimizer | AdamW, LR `3e-4`, weight decay `1e-4` |

Executed notebooks and loss plots:

- `partb-02a-byol-pretrain output.ipynb`
- `partb-03a-ijepa-pretrain output .ipynb`
- `__results___10_0 (1).png` (BYOL loss and EMA schedule)
- `__results___10_0.png` (I-JEPA loss and EMA schedule)

## Run Configuration

### BYOL

```python
method = "byol"
yolo_model = "yolo12m.yaml"
epochs = 30
batch_size = 128          # per GPU
image_size = 224
learning_rate = 3e-4
min_learning_rate = 3e-6
weight_decay = 1e-4
warmup_epochs = 1
projection_dim = 128
hidden_dim = 512
momentum = 0.99
final_momentum = 1.0
amp = True
```

### I-JEPA

```python
method = "ijepa"
yolo_model = "yolo12m.yaml"
epochs = 30
batch_size = 128          # per GPU
image_size = 224
learning_rate = 3e-4
min_learning_rate = 3e-6
weight_decay = 1e-4
warmup_epochs = 2
projection_dim = 256
momentum = 0.996
final_momentum = 1.0
num_target_blocks = 4
target_scale_min = 0.10
target_scale_max = 0.25
target_aspect_min = 0.75
target_aspect_max = 1.50
predictor_depth = 2
predictor_heads = 4
amp = True
```

These values satisfy `PretrainConfig.validate()`. Both runs completed all 30
epochs and produced 100% backbone transfer coverage, so this is not an OOM,
path, checkpoint-shape, or YOLOv12 adapter failure.

## Observed Behavior

### BYOL

The loss falls from `0.68028` in epoch 1 to `0.00106` in epoch 2 and
`0.000112` in epoch 3. It then rises gradually, reaching `0.02207` at epoch
30. The library selects epoch 3 as `best_ssl.pt` because it has the lowest
loss.

The post-training backbone diagnostic reports:

```text
diagnostic images     500
feature dimensions    512
mean L2 norm          1.000000
mean feature std      0.018106
```

This is not conclusive proof that the backbone is constant, but an objective
near zero after two epochs from random initialization is not the expected
stable BYOL trajectory. It is a strong collapse warning.

### I-JEPA

The loss falls from `0.29092` in epoch 1 to `0.00209` in epoch 2. It rises
around epochs 4-6 and then remains approximately `0.008-0.014`.

The post-training backbone diagnostic reports:

```text
diagnostic images     500
feature dimensions    512
mean L2 norm          1.000000
mean feature std      0.000059
```

This confirms representation collapse for the exported `best_ssl.pt`:
distinct images have almost identical normalized encoder features.

## Confirmed Defect 1: I-JEPA Erases Target Position Information

### Source Evidence

In `src/ssldet/methods/ijepa.py`:

```python
# Lines 73-76
def forward(self, tokens: torch.Tensor, target_union: torch.Tensor) -> torch.Tensor:
    mask_tokens = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
    predictor_input = torch.where(target_union.unsqueeze(-1), mask_tokens, tokens)
    return self.transformer(predictor_input)
```

The caller adds position before this replacement:

```python
# Line 138
predictions = self.predictor(context_tokens + position, target_union)
```

For every masked target location, `torch.where(...)` replaces the entire
`context_tokens + position` vector with the same raw `mask_token`. All target
locations therefore lose their positional encoding.

### Why This Is Incorrect

A transformer without a position signal cannot distinguish a target at one
spatial location from a target at another. Visible tokens retain position, but
every masked query token has the same initial embedding. The model cannot
learn location-specific latent prediction, which is the core I-JEPA task.

This is especially harmful for a 7 x 7 YOLO feature grid: four sampled target
blocks create multiple target positions that are all represented by the same
mask token before self-attention.

### Minimal Patch

Preserve or add position after replacing target content:

```python
class LatentPredictor(nn.Module):
    def forward(
        self,
        tokens: torch.Tensor,
        target_union: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        predictor_input = torch.where(target_union.unsqueeze(-1), mask_tokens, tokens)
        return self.transformer(predictor_input + position)

# IJEPA.forward
predictions = self.predictor(context_tokens, target_union, position)
```

### Required Regression Test

Add a test that creates at least two masked positions and verifies their
predictor input differs due to positional encoding. Existing
`tests/test_ijepa_masks.py` tests only mask shape/non-emptiness and positional
encoding shape; it does not test that masked tokens retain position.

## Confirmed Defect 2: AMP-Skipped Updates Still Advance EMA and Schedules

### Source Evidence

`src/ssldet/trainer.py:196-204`:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(method.parameters(), config.gradient_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
method.after_optimizer_step()
scheduler.step()
global_step += 1
```

`GradScaler.step(optimizer)` can skip `optimizer.step()` when gradients are
non-finite. In that case the code still:

- EMA-updates the BYOL/I-JEPA target with unchanged online weights.
- Advances the learning-rate scheduler.
- Advances the momentum schedule through `global_step`.

The executed BYOL log emits PyTorch's warning:

```text
Detected call of `lr_scheduler.step()` before `optimizer.step()`.
```

This is the expected warning when AMP skipped an optimizer step but the
scheduler was still advanced.

### Minimal Patch

Use the scale before and after `GradScaler.update()` to detect an overflow
skip. `optimizer.step()` normally returns `None`, so do not use the return
value of `scaler.step()` as the success flag.

```python
scale_before = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
step_was_skipped = scaler.get_scale() < scale_before
optimizer.zero_grad(set_to_none=True)

if not step_was_skipped:
    method.after_optimizer_step()
    scheduler.step()
    global_step += 1
```

The trainer should also log the skipped-update count per epoch.

## Checkpoint-Selection Weakness

`src/ssldet/trainer.py:237-239` saves `best_ssl.pt` whenever scalar loss is
lowest:

```python
if mean_loss < best_loss:
    best_loss = mean_loss
    torch.save(state, best_checkpoint)
```

This is reasonable only if the objective cannot collapse. In these runs, the
minimum-loss checkpoints are precisely the degenerate ones:

- BYOL: epoch 3, loss `0.000112`
- I-JEPA: epoch 2, loss `0.002087`

The downstream helper correctly transfers only `online_encoder.*`, but it
cannot know whether that encoder has collapsed. The library should not claim
lowest SSL loss is automatically the best transferable representation for
non-contrastive methods.

Recommended changes:

1. Log representation variance and mean pairwise cosine similarity on a fixed
   image-only probe sample each epoch.
2. Reject a checkpoint if variance is below a configurable threshold relative
   to an initialization baseline.
3. Record both `best_ssl.pt` and `last_ssl.pt`, but require a non-collapse
   check before downstream handoff.

## BYOL Robustness Concern: Collapse-Prone Projection Heads

This is a design concern supported by the run, not a claim of a single
confirmed source-code typo.

`src/ssldet/methods/common.py:14-19` defines the projector/predictor heads as:

```python
nn.Linear(input_dim, hidden_dim, bias=False),
nn.LayerNorm(hidden_dim),
nn.GELU(),
nn.Linear(hidden_dim, output_dim),
```

The BYOL run uses `momentum=0.99`, not the slower `0.996` default, yet its
loss still reaches `0.000112` by epoch 3. Therefore changing momentum alone
does not explain or fix the trajectory.

The implementation differs from common BYOL reference heads, which use
BatchNorm in the projection/prediction MLPs. This library also has no
variance/covariance regularizer and no collapse detector. A constant output
of the online projector/predictor and EMA target projector minimizes the
negative-cosine objective, so head collapse is not explicitly prevented.

Requested author guidance:

1. Is LayerNorm-only BYOL intentional and validated for random CNN encoders?
2. Is there a reference run showing non-collapsed feature variance for this
   implementation on a YOLO backbone?
3. Should the heads use the original BatchNorm-style BYOL MLP, with compatible
   target BatchNorm buffer handling?

The recommended validation is an A/B experiment with all seeds, data,
augmentations, and optimizer settings fixed, comparing the present LayerNorm
head against the intended reference head. Record per-epoch loss, feature
variance, pairwise cosine similarity, and downstream detection mAP.

## Additional DDP Risk

`trainer.py:139-144` wraps the method with
`DistributedDataParallel(..., broadcast_buffers=False)`. Then
`methods/common.py:36-37` copies local online buffers directly into each
rank's target model:

```python
for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
    target_buffer.copy_(online_buffer)
```

YOLO backbones contain BatchNorm running statistics. Because each rank sees a
different image shard, online and target BatchNorm buffers can diverge across
ranks. This is not proven to be the primary collapse cause, but it should be
reviewed after the confirmed defects are fixed.

## Test Coverage Gap

The repository contains `tests/test_ijepa_masks.py`, but it checks only:

- target-mask shape
- target-mask non-emptiness
- sinusoidal-position tensor shape

There is no end-to-end BYOL test, no end-to-end I-JEPA optimization test, and
no representation-collapse regression test. A successful test suite can
therefore miss the behavior reported here.

Suggested minimum tests:

1. BYOL and I-JEPA forward/backward/EMA test on a tiny spatial encoder.
2. I-JEPA test proving masked tokens retain unique positions.
3. AMP overflow test proving EMA, scheduler, and global step do not advance
   when the optimizer update is skipped.
4. Multi-step toy-data test that verifies representation variance remains
   non-trivial.
5. DDP BatchNorm-buffer consistency test, if BatchNorm remains in the student.

## Notebook Assessment

The notebooks correctly use the public course-library workflow:

- `PretrainConfig(...).validate()`
- `launch_distributed_pretrain(..., num_processes=2)`
- random `yolo12m.yaml` initialization
- SSL-pool-only image paths
- required 30-epoch protocol
- correct backbone transfer coverage checks

The configurations are not the cause of I-JEPA's position-token defect or the
trainer's AMP update defect.

However, the notebooks are not "perfect" in a diagnostic sense. They trusted
the library's `best_ssl.pt` policy and only measured feature variation after
training. They should add an image-only anti-collapse guard before publishing
or transferring a checkpoint. This omission did not create the collapse; it
allowed the collapsed checkpoint to be selected and exported without stopping
the workflow.

## Requested Resolution

Please confirm whether the maintainers prefer:

1. A patched library release implementing the I-JEPA position fix and AMP
   skipped-step guard, followed by rerunning the affected notebooks; or
2. Written approval for students to apply a minimal local patch, publish the
   patched source version, and document the exact commit/version in notebooks.

Until then, the current I-JEPA `best_ssl.pt` must not be interpreted as a
valid pretrained representation, and the BYOL result should be treated as
suspect pending a non-collapse diagnostic or patched reference implementation.
