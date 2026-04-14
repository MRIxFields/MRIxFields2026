# Submission Guidelines

> **Note**: This document is subject to change. Please check back for updates.

## Validation Phase (To be updated)

Submit your predictions as a **zip file** (`submission_pack.zip`) via the [Synapse platform](https://www.synapse.org/Synapse:syn72060672).

### Directory Structure

```
submission_pack/
├── task1/                              # Task 1: Any -> 7T
│   ├── 0.1T_to_7T/
│   │   ├── pred/
│   │   │   ├── T1W/
│   │   │   │   └── 0.1T_001.nii.gz
│   │   │   ├── T2W/
│   │   │   └── T2FLAIR/
│   │   └── seg/                        # Optional
│   │       ├── T1W/
│   │       │   └── 0.1T_001_seg.nii.gz
│   │       ├── T2W/
│   │       └── T2FLAIR/
│   ├── 1.5T_to_7T/
│   ├── 3T_to_7T/
│   └── 5T_to_7T/
├── task2/                              # Task 2: 0.1T -> Higher
│   ├── 0.1T_to_1.5T/
│   ├── 0.1T_to_3T/
│   ├── 0.1T_to_5T/
│   └── 0.1T_to_7T/
└── task3/                              # Task 3: Any -> Any (single unified model)
    ├── 0.1T_to_1.5T/
    ├── 0.1T_to_3T/
    ├── 0.1T_to_5T/
    ├── 0.1T_to_7T/
    ├── 1.5T_to_7T/
    ├── 3T_to_7T/
    └── 5T_to_7T/
```

Each translation pair contains `pred/` (required) and `seg/` (optional, for Dice/Volume metrics). Predictions must be NIfTI (`.nii.gz`), float32, 364x436x364, intensity in [0, 1]. Filenames must match input. Segmentation files append `_seg` suffix.

You may submit for specific tasks, pairs, or modalities only. Missing predictions are not penalized.

## Testing Phase (To be updated)

*Docker container submission details will be provided here.*

## Contact

For submission issues: **mrixfields@outlook.com**
