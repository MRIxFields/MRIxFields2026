# Submission Guidelines

> **Note**: This document is subject to change. Please check back for updates.

## Validation Phase (To be updated)

Submit **one zip per task** via the [Synapse platform](https://www.synapse.org/Synapse:syn72060672):

| Zip filename | Task | Pairs |
|--------------|------|-------|
| `task1.zip` | Any → 7T | 4 (`0.1T_to_7T`, `1.5T_to_7T`, `3T_to_7T`, `5T_to_7T`) |
| `task2.zip` | 0.1T → Higher | 4 (`0.1T_to_1.5T`, `0.1T_to_3T`, `0.1T_to_5T`, `0.1T_to_7T`) |
| `task3.zip` | Any → Any (unified model) | 20 directed pairs (5 src × 4 tgt, incl. downfield) |

**Do not** bundle multiple tasks into one zip — each task is a separate Synapse upload. If you don't participate in a given task, just skip its zip.

### Directory Structure (inside each zip)

The zip filename already encodes the task, so each zip's internal layout starts at the modality level — no `task{N}/` prefix. This mirrors the dataset's `{modality}/{field}/<file>.nii.gz` ordering one-to-one (the submission's `pair` plays the role of the dataset's `field`). Filenames are identical to the corresponding GT file.

```
task1.zip                                     # Task 1: Any -> 7T  (4 pairs × 3 modalities)
└── (zip root)
    ├── T1W/
    │   ├── 0.1T_to_7T/
    │   │   ├── pred/                          # required — 17 subjects per (modality, pair)
    │   │   │   ├── P_T1W_7T_0001.nii.gz
    │   │   │   ├── P_T1W_7T_0002.nii.gz
    │   │   │   └── ...                        # one file per validation subject
    │   │   └── seg/                           # optional — Dice / Volume only; same 17 subjects
    │   │       ├── P_T1W_7T_0001_seg.nii.gz
    │   │       ├── P_T1W_7T_0002_seg.nii.gz
    │   │       └── ...
    │   ├── 1.5T_to_7T/                        # same pred/ + optional seg/ shape
    │   ├── 3T_to_7T/
    │   └── 5T_to_7T/
    ├── T2W/                                   # same 4 pairs × {pred, seg} shape as T1W
    └── T2FLAIR/

task2.zip                                     # Task 2: 0.1T -> Higher  (4 pairs × 3 modalities)
└── (zip root)
    ├── T1W/
    │   ├── 0.1T_to_1.5T/                      # same pred/ + optional seg/ shape as task1
    │   ├── 0.1T_to_3T/
    │   ├── 0.1T_to_5T/
    │   └── 0.1T_to_7T/
    ├── T2W/
    └── T2FLAIR/

task3.zip                                     # Task 3: Any -> Any  (20 pairs × 3 modalities, pred/ only)
└── (zip root)
    ├── T1W/                                   # 5 source fields × 4 target fields = 20 directed pairs (incl. downfield)
    │   ├── 0.1T_to_1.5T/
    │   │   └── pred/                          # required — 17 subjects per (modality, pair)
    │   │       ├── P_T1W_1.5T_0001.nii.gz
    │   │       ├── P_T1W_1.5T_0002.nii.gz
    │   │       └── ...
    │   ├── 0.1T_to_3T/
    │   ├── 0.1T_to_5T/
    │   ├── 0.1T_to_7T/
    │   ├── 1.5T_to_0.1T/
    │   ├── 1.5T_to_3T/
    │   ├── 1.5T_to_5T/
    │   ├── 1.5T_to_7T/
    │   ├── 3T_to_0.1T/
    │   ├── 3T_to_1.5T/
    │   ├── 3T_to_5T/
    │   ├── 3T_to_7T/
    │   ├── 5T_to_0.1T/
    │   ├── 5T_to_1.5T/
    │   ├── 5T_to_3T/
    │   ├── 5T_to_7T/
    │   ├── 7T_to_0.1T/
    │   ├── 7T_to_1.5T/
    │   ├── 7T_to_3T/
    │   └── 7T_to_5T/
    ├── T2W/
    └── T2FLAIR/
```

**Per-task contents** (validation phase has 17 paired subjects per `(modality, pair)`):

- **Task 1** — `Any → 7T`: 4 pairs × 3 modalities = **12** `(modality, pair)` directories. Each contains `pred/` (required, up to 17 NIfTI files) and `seg/` (optional, same 17 subjects, drives Dice / Volume metrics).
- **Task 2** — `0.1T → Higher`: same shape as Task 1 — 4 pairs × 3 modalities = **12** directories, each with `pred/` + optional `seg/`.
- **Task 3** — `Any → Any` (single unified model): 20 directed pairs (incl. downfield, e.g. `7T_to_0.1T`) × 3 modalities = **60** directories. Each contains `pred/` only — Task 3 is scored on voxel-level metrics (nRMSE / SSIM / LPIPS) and **does not** use segmentation. Any `seg/` placed inside `task3.zip` will be ignored.

Partial submissions are accepted: omit any modality / pair / subject you don't predict. Missing predictions are filled with worst-case metric values rather than skipped — see [evaluation-2026/README.md](evaluation-2026/README.md) "Missing-submission handling" for the exact arithmetic.

### Building the zips

This repo ships [submission_pack/](submission_pack/) as a starter template — three subtrees (`task1/`, `task2/`, `task3/`) already laid out with empty 0-byte `.nii.gz` placeholders so you see the exact filenames the scorer expects. Replace each placeholder with your model's prediction, then zip each task subtree:

```bash
# cd into the task subtree first, then zip its modality folders.
# This keeps the zip's internal root at the modality level (no `task{N}/` prefix).
cd Submission/submission_pack/task1 && zip -r ~/task1.zip T1W T2W T2FLAIR
cd Submission/submission_pack/task2 && zip -r ~/task2.zip T1W T2W T2FLAIR
cd Submission/submission_pack/task3 && zip -r ~/task3.zip T1W T2W T2FLAIR
```

> Note: the local directory name `submission_pack/` is just a working folder in this repo — it is **not** part of the submission and does **not** appear inside any uploaded zip.

### File Format & Naming

Predictions must be NIfTI (`.nii.gz`), float32, 364x436x364, intensity in [0, 1].

Filenames must match the corresponding ground-truth file:

| Type | Pattern | Example |
|------|---------|---------|
| Prediction | `P_{MOD}_{TARGET_FIELD}_{ID:04d}.nii.gz` | `P_T1W_7T_0001.nii.gz` |
| Segmentation (Task 1/2 only) | `P_{MOD}_{TARGET_FIELD}_{ID:04d}_seg.nii.gz` | `P_T1W_7T_0001_seg.nii.gz` |

- `MOD` ∈ {T1W, T2W, T2FLAIR}
- `TARGET_FIELD` is the **target** field of the pair (e.g. `7T` for `0.1T_to_7T`), not the input
- `ID` is the 4-digit zero-padded subject ID from the input scan
- Subject ID is shared across all field strengths for the same volunteer, so for a `0.1T_to_7T` prediction of subject `0001`, you transform `P_T1W_0.1T_0001.nii.gz` → `P_T1W_7T_0001.nii.gz`


## Testing Phase (To be updated)

*Docker container submission details will be provided here.*

## Contact

For submission issues: **mrixfields@outlook.com**
