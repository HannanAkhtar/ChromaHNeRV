# UVG7 HNeRV Benchmark

This benchmark trains one independent network for every sequence, model size, and architecture. It never combines videos into a shared model.

## Dataset

Arrange decoded frames as direct sequence directories:

```text
/datasets/UVG/
  Beauty/
    000001.png
    ...
  Bosphorus/
  HoneyBee/
  Jockey/
  ReadySetGo/
  ShakeNDry/
  YachtRide/
```

Pass `/datasets/UVG` to the suite launcher. When invoking `train_chroma_hnerv.py` directly, pass one sequence directory such as `/datasets/UVG/HoneyBee`, not the parent UVG directory.

The protocol expects 1080x1920 source frames and applies a 960x1920 center crop. Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, and YachtRide use 600 frames; ShakeNDry uses 300. Metadata comes from `configs/uvg7.json`.

## Protocol

The encoder and decoder stride schedule is `5 4 4 3 2`. Starting from the 2x4 latent representation, decoder stages are:

```text
10x20 -> 40x80 -> 160x320 -> 480x960 -> 960x1920
```

Compatibility aliases retain their historical names:

- `a320`: shared output is 480x960 on UVG.
- `a160`: shared output is 160x320 on UVG.
- Chroma420 CbCr output is 480x960 before upsampling.

Each sequence-size pair trains ten configurations: Full RGB, Full YCbCr444, and RGBSplit/Chroma420 at a320/a160 with widths 8 and 4. Across seven sequences and four model sizes this is 280 runs. Final evaluation remains uniform M8/E6 and records both full-precision and quantized metrics.

## Environment

For a modern Python and RTX 4500 Ada environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_modern.txt
```

`torch==2.5.1` and `torchvision==0.20.1` are a matched pair. Install them through the package source appropriate for the workstation CUDA driver; this repository does not hard-code a CUDA wheel URL. `requirements.txt` remains untouched, and `requirements_legacy.txt` records the original environment.

## Commands

Dry-run the complete matrix:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /datasets/UVG \
  --output-root /runs/uvg7_hnerv_150e \
  --dry-run
```

Run the five-job smoke suite:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /datasets/UVG \
  --output-root /runs/uvg7_hnerv_150e \
  --smoke
```

Run the full sequential suite on GPU 0:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /datasets/UVG \
  --output-root /runs/uvg7_hnerv_150e \
  --backup-root /persistent-backup/uvg7 \
  --gpu 0
```

Completed jobs are skipped. A directory containing `model_latest.pth` without a complete `completion.json` resumes automatically with the same command. Destructive reruns require both `--force --yes`.

Run one sequence, size, or family:

```bash
python run_uvg_hnerv_suite.py \
  --data-root /datasets/UVG --output-root /runs/uvg7 \
  --sequences HoneyBee --sizes 1.5 --families chroma420_a160 --widths 4
```

Estimate runtime by running the smoke suite or a one-epoch restricted command, then scale using the observed epoch time. Full runtime depends strongly on architecture and evaluation cost.

Merge per-run results:

```bash
python merge_uvg_results.py \
  --output-root /runs/uvg7_hnerv_150e \
  --destination /runs/uvg7_results
```

The merger writes `uvg7_all_runs.csv`, `uvg7_missing_runs.csv`, `uvg7_duplicate_runs.csv`, and `uvg7_failed_runs.csv`. It accepts only complete M8/E6 final rows and preserves original and quantized metric columns.
