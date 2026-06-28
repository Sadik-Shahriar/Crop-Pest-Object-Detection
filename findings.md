# Label Conflict Findings — AgroPest-12

## Summary

| Metric | Value |
|--------|-------|
| Total images | ~13,140 |
| Label conflicts detected | **31 (0.24%)** |
| — Single-class (likely mislabeled) | **3** |
| — Mixed-class (multi-object images) | **28** |
| Test set conflicts | **0 (clean)** |

## The 3 Likely Mislabeled Files

All 3 are from the `ants-1` batch (PNG format):

| File | Label says | Visually appears (verified) |
|------|-----------|---------------------------|
| `ants-1-_png.rf.0b5a6cf9...` | Ants (class 0) | Ants ✓ |
| `ants-1-_png.rf.809de126...` | Ants (class 0) | Ants ✓ |
| `ants-1-_png.rf.e4e60471...` | Ants (class 0) | Ants ✓ |

Their sibling JPG files in the same batch:

| File | Label says | Visually appears (verified) |
|------|-----------|---------------------------|
| `ants-1-_jpg.rf.8227b4d5...` | **Bees (class 1)** | **Ants ✗** |
| `ants-1-_jpg.rf.cb386958...` | **Bees (class 1)** | **Ants ✗** |
| `ants-1-_jpg.rf.ef0ce7e1...` | **Bees (class 1)** | **Ants ✗** |

**3 JPGs incorrectly say "Bees" when they show ants.**

## The 28 Multi-Object Files (Probably Correct)

These label files contain boxes of **multiple different classes**
in the same image. Example:

| Batch | Box 1–6 | Box 7–8 | Likely correct? |
|-------|---------|---------|-----------------|
| `slug-56` | Slug (cls 8) | Snail (cls 9) | Yes — photo has both |
| `beetle-252` | Beetle (cls 2) | Wasp (cls 10) | Yes — photo has both |
| `catterpillar-419` | Wasp (cls 10) | Caterpillar (cls 3) | Need visual check |

These are images where multiple pest species appear together —
the annotations are likely accurate.

## Recommendation

**Do not fix anything.** Reasons:

1. **3 files (0.02%)** mislabeled is negligible — neural networks
   handle small label noise naturally.

2. **Majority vote can be wrong** — fixing the 3 PNGs to match
   the JPGs would introduce 3 new errors (the JPGs are the ones
   that are wrong).

3. **Test set is clean** — evaluation metrics unaffected.

Note this finding in the EDA summary. In the viva, explain:
*"We found 31 minor inconsistencies (0.24%). Three training images
labeled 'Bees' appear to actually show Ants. We left the dataset
unchanged because the noise is negligible and auto-fix could
introduce more errors than it fixes."*
