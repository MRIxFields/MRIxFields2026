# MRIxFields2026 — Tutorial

Step-by-step Jupyter notebooks for getting started with the MRIxFields2026 challenge.

## Scope

> **This Tutorial demonstrates a Task 1 entry-level example** — single-pair translation (0.1T → 7T, T1W) using the CUT method, with all model / loss / dataset code defined **inline** so you can read the pipeline end-to-end in one place.
>
> For full Baseline coverage — **all 3 tasks**, **5 field strengths**, **3 modalities**, **3 methods** (CUT / CycleGAN / StarGAN v2), with DDP, 51 YAML configs, and 3 training modes — use [`Baseline/`](../Baseline/README.md):
> - **Task 1** (any → 7T): [`Baseline/configs/task1/`](../Baseline/configs/task1/) — 24 configs
> - **Task 2** (0.1T → any higher field): [`Baseline/configs/task2/`](../Baseline/configs/task2/) — 24 configs
> - **Task 3** (any → any, single StarGAN v2 model with 15 joint modality–field domains): [`Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml`](../Baseline/configs/task3/stargan/any_to_any_all_modalities.yaml)
>
> The inline interfaces in this Tutorial mirror [`Baseline/mrixfields/`](../Baseline/mrixfields/) (`ResnetGenerator`, `NLayerDiscriminator`, `PatchSampleF`, `GANLoss`, `PatchNCELoss`, `PerceptualLoss`, `CUTModel`). Each notebook also points to the corresponding `Baseline/` files and CLI commands for the production version.

## Prerequisites

Ensure you have set up the environment. See [main README §3](../README.md#3-set-up-environment).

- Data placed in `$DATA_DIR` (configured in `.env`)
- SynthSeg installed at `$SYNTHSEG_DIR` (required by notebook 03 for Dice/Volume metrics, see [Evaluation/README.md](../Evaluation/README.md))

## Getting Started

```bash
conda activate mf
cd Tutorial
jupyter notebook
```

Or run them headless in order:

```bash
conda activate mf
cd Tutorial
jupyter nbconvert --to notebook --execute --inplace 01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_evaluation.ipynb
```

## Notebooks

Notebooks are designed to run sequentially — each one builds on the outputs of the previous.

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [01_data_exploration.ipynb](01_data_exploration.ipynb) | Load NIfTI volumes, inspect headers/shapes (364×436×364), filename convention (`R_*` vs `P_*`), 4 data splits, visualize across 5 field strengths and 3 modalities |
| 02 | [02_baseline.ipynb](02_baseline.ipynb) | Self-contained inline CUT baseline (0.1T → 7T, T1W): preprocessing (slices 72–291) → inline model/loss definitions → pretrain (5 epochs, retro unpaired) → finetune (2 epochs, pro paired) → inference on subject 0007 → save prediction `P_T1W_7T_0007.nii.gz` to `tmp/inference/` |
| 03 | [03_evaluation.ipynb](03_evaluation.ipynb) | Evaluate predictions from notebook 02: voxel metrics (nRMSE, SSIM, LPIPS) — required by all 3 tasks; SynthSeg segmentation + DGM overlay + Dice/Volume — Task 1/2 only; full submission flow (segment → build → zip → upload) for Task 1/2 and Task 3 |

## Intermediate Files

Notebooks 02 and 03 write intermediate results to `tmp/`:

```
tmp/
├── preprocessed/                              # 2D axial slices (NPZ) from notebook 02
│   ├── retro_train/T1W/{0.1T,7T}/...          #   220 slices/volume (z=72..291)
│   └── pro_train/T1W/{0.1T,7T}/...
├── pretrain_generator.pth                     # CUT pretrained checkpoint (notebook 02 stage 1)
├── finetune_generator.pth                     # Finetuned checkpoint (notebook 02 stage 2)
├── inference/
│   ├── P_T1W_7T_0007.nii.gz                   # Prediction — target-field naming, NO `_pred` suffix
│   └── eval_results.csv                       # Per-subject metrics (notebook 03)
└── seg/                                       # SynthSeg segmentations from notebook 03 (Task 1/2 only)
    ├── pred/P_T1W_7T_0007_seg.nii.gz          #   prediction segmentation
    └── target/P_T1W_7T_0007_seg.nii.gz        #   ground-truth segmentation
```

> **Filename note.** The Tutorial saves predictions with **target-field naming** (`P_T1W_7T_0007.nii.gz`, no `_pred` suffix) — this matches the final submission convention enforced by [`Submission/build_submission/build_submission.py`](../Submission/build_submission/build_submission.py). Production [`Baseline/scripts/inference.py`](../Baseline/scripts/inference.py) instead preserves the *source* filename (`P_T1W_0.1T_0007.nii.gz`) and lets `build_submission.py` rename to the target field at packing time. Notebook 03 matches predictions to targets by trailing subject ID, so either convention works for evaluation.

For data split details, see the [main README §2](../README.md#2-get-the-data).
