# ChromaNeRV: HNeRV Implementation

## Anonymous Supplementary Code

This repository accompanies an anonymous paper submission. It extends the original HNeRV implementation with luma-chroma decoder variants, matched RGB controls, quantized evaluation, and reproducible multi-sequence launchers.

## Overview

- **Full RGB** reconstructs RGB directly at full resolution.
- **Full YCbCr444** predicts Y, Cb, and Cr at full resolution.
- **RGBSplit** keeps a shared decoder and narrows the high-resolution RGB tail.
- **ChromaNeRV** uses the narrow full-resolution path for luminance, predicts CbCr at half resolution, and bilinearly upsamples chroma.
- All YCbCr variants reconstruct RGB with the inverse full-range BT.709 transform before RGB-domain metrics are computed.

## Repository Structure

```text
train_chroma_hnerv.py   training, evaluation, quantization, and rate reporting
model_chroma_hnerv.py   RGBSplit and ChromaNeRV model definitions
model_all.py            upstream HNeRV model and frame dataset
hnerv_utils.py          losses, metrics, colour conversion, and quantization helpers
run_uvg_hnerv_suite.py  resumable UVG7 experiment launcher
uvg_utils.py            UVG presets, job matrix, and backup restoration
merge_uvg_results.py    completed-run result merger
efficient_nvloader.py   quantized checkpoint playback utility
configs/uvg7.json       UVG sequence metadata
tests/                  automated tests
```

Datasets, trained checkpoints, generated images, videos, logs, and result files are not included.

## Installation

Create one recommended environment:

```bash
conda create -n chromahnerv python=3.10
conda activate chromahnerv
python -m pip install --upgrade pip
```

Install a mutually compatible PyTorch and torchvision build for the CUDA driver and GPU on the target machine. Follow the official PyTorch installation selector; this repository does not hard-code a CUDA wheel URL. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

LPIPS is optional and is loaded lazily. Install `lpips` to report `lpips_alex`; otherwise the trainer prints an explanatory message and records `NaN`. DISTS, VMAF, FID, and KID are not computed by the included pipeline and require separate external tools if needed.

`requirements_legacy.txt` records the historical upstream environment and is not recommended for new installations.

## Dataset Preparation

Use naturally sortable frame names and direct per-sequence directories:

```text
data/
|-- bunny/
|   |-- 000000.png
|   `-- ...
`-- uvg/
    |-- Beauty/
    |-- Bosphorus/
    |-- HoneyBee/
    |-- Jockey/
    |-- ReadySetGo/
    |-- ShakeNDry/
    `-- YachtRide/
```

The dataset loader accepts PNG, JPG, and JPEG files, ignores hidden and nested files, and sorts frames naturally. UVG source frames are expected at 1080x1920 and the UVG preset applies the existing 960x1920 centre crop.

The supplied paper protocol description specifies the first 132 frames. The current checked-in UVG7 launcher, however, validates and processes the complete manifest lengths: 600 frames for all listed UVG sequences except ShakeNDry, which uses 300. No implicit 132-frame truncation exists. This release preserves that behavior; see **Reproducibility Notes** before comparing against a 132-frame paper protocol. Bunny experiments use a prepared 132-frame directory.

## Experiment Configurations

The paper grid contains ten configurations:

1. Full RGB
2. Full YCbCr444
3. RGBSplit-A320-W8
4. ChromaNeRV-A320-W8
5. RGBSplit-A320-W4
6. ChromaNeRV-A320-W4
7. RGBSplit-A160-W8
8. ChromaNeRV-A160-W8
9. RGBSplit-A160-W4
10. ChromaNeRV-A160-W4

Model-size targets are `0.35`, `0.75`, `1.5`, and `3.0` million parameters. Each sequence, size, and configuration is trained independently. Split models use fixed branch widths; A320 and A160 are compatibility aliases for the selected decoder split stages.

## Dry Run

Print the complete UVG7 grid without accessing the dataset or training:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /path/to/uvg \
  --output-root output/uvg7 \
  --dry-run
```

The default grid contains exactly 280 jobs: seven sequences x four model sizes x ten configurations.

## Running the UVG7 Experiments

Run the complete sequential grid on GPU 0:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /path/to/uvg \
  --output-root output/uvg7 \
  --gpu 0
```

Completed jobs are skipped. An incomplete directory containing `model_latest.pth` resumes automatically. A destructive rerun requires `--force --yes`.

Persistent backup storage is optional:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /path/to/uvg \
  --output-root output/uvg7 \
  --backup-root /path/to/persistent/backup \
  --strict-backup \
  --gpu 0
```

With `--backup-root`, valid missing or newer artifacts are restored before a job is classified as complete, resumable, or new. Without `--strict-backup`, backup failures are warnings rather than training failures.

## Running a Selected Experiment

```bash
python run_uvg_hnerv_suite.py \
  --data-root /path/to/uvg \
  --output-root output/uvg7 \
  --sequences HoneyBee \
  --sizes 1.5 \
  --families chroma420_a320 \
  --widths 8 \
  --gpu 0
```

Available families are `rgb444`, `ycbcr444`, `rgbsplit_a320`, `chroma420_a320`, `rgbsplit_a160`, and `chroma420_a160`. `--widths` applies only to split families.

## Running Bunny Experiments

Bunny uses a 640x1280 crop and decoder strides `5 4 4 2 2`, while UVG uses a 960x1920 crop and decoder strides `5 4 4 3 2`. The following direct command is the shared base for a 150-epoch Bunny run:

```bash
python train_chroma_hnerv.py \
  --data_path /path/to/bunny --vid bunny --run_dir output/bunny/rgb444_1p5 \
  --experiment rgb444_hnerv --modelsize 1.5 \
  --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 \
  --enc_dim 64_16 --ks 0_1_5 --reduce 1.2 --lower_width 12 \
  --crop_list 640_1280 --resize_list -1 --loss L2 \
  --conv_type convnext pshuffel --act gelu --norm none \
  --batchSize 2 --epochs 150 --eval_freq 30 --manualSeed 1 \
  --quant_model_bit 8 --quant_embed_bit 6
```

Select the other families by replacing or appending the architecture arguments:

| Configuration | Arguments |
| --- | --- |
| Full RGB | `--experiment rgb444_hnerv` |
| Full YCbCr444 | `--experiment ycbcr444_hnerv` |
| RGBSplit-A320-W8/W4 | `--experiment rgbsplit_a320 --split_stage a320 --branch_width_mode fixed --branch_width 8` (or `4`) |
| ChromaNeRV-A320-W8/W4 | `--experiment chroma420_a320 --split_stage a320 --branch_width_mode fixed --branch_width 8` (or `4`) |
| RGBSplit-A160-W8/W4 | `--experiment rgbsplit_a160 --split_stage a160 --branch_width_mode fixed --branch_width 8` (or `4`) |
| ChromaNeRV-A160-W8/W4 | `--experiment chroma420_a160 --split_stage a160 --branch_width_mode fixed --branch_width 8` (or `4`) |

Repeat with `--modelsize 0.35`, `0.75`, `1.5`, and `3.0` as required.

## Evaluation

Evaluate a checkpoint in full precision by disabling PTQ explicitly and reproducing its training architecture:

```bash
python train_chroma_hnerv.py \
  --data_path /path/to/sequence --vid sequence \
  --run_dir output/evaluation \
  --dataset_preset uvg_hnerv \
  --experiment chroma420_a320 --modelsize 1.5 \
  --split_stage a320 --branch_width_mode fixed --branch_width 8 \
  --enc_strds 5 4 4 3 2 --dec_strds 5 4 4 3 2 \
  --crop_list 960_1920 --resize_list -1 \
  --eval_only --not_resume --weight /path/to/checkpoint/epoch150.pth \
  --quant_model_bit -1 --quant_embed_bit -1
```

Add `--dump_images`, `--dump_videos`, or both for visual outputs. Evaluation CSVs include full-precision quality metrics, parameter counts, estimated GFLOPs, and optional timing fields. FPS measurement is enabled with `--eval_fps` or `--measure_latency`.

## Quantization

The main quantized evaluation is M8/E6: 8-bit model weights and 6-bit frame embeddings. Use the same architecture and checkpoint command as above with:

```text
--quant_model_bit 8 --quant_embed_bit 6 --final_eval_mode full
```

For example:

```bash
python train_chroma_hnerv.py \
  --data_path /path/to/sequence --vid sequence \
  --run_dir output/quantized_evaluation \
  --dataset_preset uvg_hnerv \
  --experiment chroma420_a320 --modelsize 1.5 \
  --split_stage a320 --branch_width_mode fixed --branch_width 8 \
  --enc_strds 5 4 4 3 2 --dec_strds 5 4 4 3 2 \
  --crop_list 960_1920 --resize_list -1 \
  --eval_only --not_resume --weight /path/to/checkpoint/epoch150.pth \
  --quant_model_bit 8 --quant_embed_bit 6 --final_eval_mode full
```

The final CSV reports fixed-width and Huffman-estimated rates together with full-precision and quantized metrics. `quant_vid.pth` stores the packed quantized representation when quantization is enabled.

## Rate and Result Summaries

Merge complete M8/E6 UVG runs:

```bash
python merge_uvg_results.py \
  --output-root output/uvg7 \
  --destination results/uvg7
```

This writes `uvg7_all_runs.csv`, `uvg7_missing_runs.csv`, `uvg7_duplicate_runs.csv`, and `uvg7_failed_runs.csv`. Coded rates are computed by `train_chroma_hnerv.py` and stored in each final CSV, including `fixed_width_bpp` and Huffman fields. The cleaned main pipeline does not include a generic BD-rate command; any BD-rate values must be computed from the merged RD points with the paper's stated interpolation protocol.

## Output Structure

A completed suite run contains:

```text
command.txt
config.json
environment.txt
git_commit.txt
rank0.txt
model_latest.pth
model_best.pth
epoch150.pth
epoch150.csv
completion.json
quant_vid.pth
img_decoder.pth       # exported for compatible full HNeRV models
```

Launcher logs are stored under `<output-root>/logs/`. The final CSV contains aggregate and frame-distribution statistics; the current trainer does not write a separate per-frame metrics CSV.

Decode selected frames from compatible packed full-HNeRV artifacts with:

```bash
python efficient_nvloader.py \
  --decoder /path/to/img_decoder.pth \
  --ckt /path/to/quant_vid.pth \
  --dump_dir output/decoded --frames 16
```

## Tests

Run the automated suite:

```bash
python -m pytest -q
```

Print the five-job one-epoch smoke plan or execute it by omitting `--dry-run`:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /path/to/uvg \
  --output-root output/uvg7 \
  --smoke --dry-run
```

The executable smoke test requires the complete UVG frame directories and a CUDA-capable PyTorch environment.

## Reproducibility Notes

- Default seed: 1.
- Training duration: 150 epochs.
- One independent model is trained per sequence, size, and configuration.
- Split experiments use fixed W8/W4 branches at A320/A160.
- Final suite evaluation uses M8/E6 and the final checkpoint.
- The Bunny data convention is 132 frames.
- **Protocol inconsistency:** the supplied paper description says the first 132 frames of every sequence, but `configs/uvg7.json` and the current launcher enforce full UVG lengths of 600/300 and train on every discovered frame. This was not altered during anonymization.
- Quality values can vary slightly across PyTorch, CUDA, and GPU versions. FPS is hardware dependent.

## Upstream HNeRV

This code is based on the original HNeRV implementation:

```bibtex
@inproceedings{chen2023hnerv,
  title={{HN}e{RV}: A Hybrid Neural Representation for Videos},
  author={Hao Chen and Matthew Gwilliam and Ser-Nam Lim and Abhinav Shrivastava},
  booktitle={CVPR},
  year={2023}
}
```

## Anonymous Paper Citation

```bibtex
@inproceedings{anonymous2027chromanerv,
  title={ChromaNeRV: Luma--Chroma Capacity Allocation for Efficient
         Neural Video Representation},
  author={Anonymous},
  booktitle={Under Review},
  year={2027}
}
```
