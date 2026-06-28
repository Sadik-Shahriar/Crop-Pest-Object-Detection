# Explanation: What Happened with the Labels?

## The Problem in Simple Terms

Imagine someone took 13,140 photos of insects and put them into labeled
folders. Most are correct, but some photos ended up in the wrong folders.
Our job was to find which ones are misplaced.

## Why This Matters for Training

If a model sees a photo labeled "Bees" but it actually shows an Ant,
the model learns the wrong thing. It starts associating ant features
with the word "Bees." This hurts accuracy during testing.

## How We Detected the Problem

We could not look at every photo individually (13,000+ photos).
Instead, we used a computer check:

**Step 1 — Group by batch name**

The filenames look like:
```
ants-1-_jpg.rf.a1b2c3d4e5f6.txt
ants-1-_jpg.rf.f6e5d4c3b2a1.txt
slug-56-_jpg.rf.x1y2z3w4v5u6.txt
```

The part before `.rf.` is the upload batch name — the folder the
person uploaded their photos into on Roboflow. Images in the same
batch likely contain the same insect.

**Step 2 — Majority vote**

For each batch name (e.g., `ants-1`), we looked at ALL the label
files. The most common class_id among them is the "majority."

| Batch | Total files | Class 0 boxes | Class 1 boxes | Majority |
|-------|-------------|---------------|---------------|----------|
| ants-1 | 6 files | 3 PNG files say class 0 | 3 JPG files say class 1 | Class 1 (Bees) |

The majority class for `ants-1` is 1 (Bees). Most files in this
batch say Bees.

**Step 3 — Find the outliers**

Any file whose class_id disagrees with its batch's majority is a
"conflict."

But there are two kinds:

### Type 1: Single-class mismatches (likely wrong)

A file where **100% of its boxes disagree** with the batch majority.

```
Example — ants-1-_png.xxx.txt:
  Line 1: 0 0.45 0.46 0.72 0.67   ← says class 0 (Ants)
  All boxes: class 0

  But the rest of the ants-1 batch says class 1 (Bees).
```

This is suspicious. The file is entirely "Ants" while its siblings
are "Bees." Something is off.

**The 3 safe-to-fix files are this type.**

### Type 2: Mixed-class files (probably fine)

A file where **some boxes agree and some disagree** with the majority.

```
Example — slug-56-_jpg.xxx.txt:
  Line 1: 8 0.30 0.40 0.20 0.15   ← class 8 (Slug)  ✓ agrees with batch
  Line 2: 8 0.50 0.50 0.15 0.15   ← class 8 (Slug)  ✓ agrees
  Line 3: 8 0.70 0.60 0.10 0.10   ← class 8 (Slug)  ✓ agrees
  Line 4: 9 0.40 0.30 0.05 0.05   ← class 9 (Snail) ✗ disagrees
  Many class 8 boxes + a few class 9 boxes
```

This looks like a photo that genuinely has **both** a slug and a
snail in it. The minority label (Snail) is likely correct.

**The 28 needs-review files are this type.**

## The Important Caveat

The majority vote is NOT proof that the majority is right. It is
only a statistical guess.

Consider the `ants-1` batch:

| Version | File | Label says | Shows (per your observation) |
|---------|------|------------|------------------------------|
| JPG #1  | ants-1-_jpg.rf.8227... | Bees (class 1) | **Ant** |
| JPG #2  | ants-1-_jpg.rf.cb38... | Bees (class 1) | **Ant** |
| JPG #3  | ants-1-_jpg.rf.ef0c... | Bees (class 1) | **Ant** |
| PNG #1  | ants-1-_png.rf.0b5a... | Ants (class 0) | **Ant** |
| PNG #2  | ants-1-_png.rf.809d... | Ants (class 0) | **Ant** |
| PNG #3  | ants-1-_png.rf.e4e6... | Ants (class 0) | **Ant** |

**All 6 files show ants.** The 3 JPGs incorrectly say "Bees."
The 3 PNGs correctly say "Ants."

Under majority vote: class 1 (Bees) is majority → our "fix" would
change the 3 correct PNGs to the wrong label! This is why
**APPLY_FIXES should stay False** — we cannot trust the majority
without visual verification.

## What You Saw

In the annotated samples, your code correctly read the label file.
The label file said class_id=1 (Bees). Your code displayed "Bees."
But the photo showed an ant. The code was reading what the label
file said — the label file itself was wrong. That was not a code
bug. It was a **dataset label error**.

## Bottom Line

- The code is correct — it reads class_id from .txt and maps via data.yaml
- Roughly 31 out of 13,140 label files (0.24%) appear inconsistent
- Some are genuine mislabeling (the ants-1 batch)
- Some are correct multi-class annotations (slug-56 has both slugs and snails)
- Do NOT auto-fix — the majority vote is not reliable enough
- The test set has 0 conflicts — evaluation metrics are trustworthy
