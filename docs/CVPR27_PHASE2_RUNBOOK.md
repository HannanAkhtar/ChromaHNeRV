# CVPR 2027 Phase 2

Checkout in Colab:

```bash
git clone https://github.com/HannanAkhtar/ChromaHNeRV.git
cd ChromaHNeRV
git checkout experiment/cvpr27-phase2-controls-latency
git checkout 01ad0dc847ac31e03f9da40e69ca981bf305f366
git rev-parse HEAD
```

Open `notebooks/ChromaNeRV_CVPR_Phase2_Training_and_Latency.ipynb` in an A100 Colab runtime. Run all cells once with `RUN_TRAINING=False` and `RUN_LATENCY=False` for installation, archive verification, tests, and the 16-job plan. The notebook finds `ChromaNeRV_UVG132.zip` and `ChromaNeRV_CVPR_Checkpoints_20261115_021417.zip` in MyDrive, copies them to `/content`, verifies ZIP integrity, and extracts them locally. Provide the original Bunny ZIP and original Bunny `command.txt` separately. A missing Bunny input is an error, never replaced by UVG.

Set `RUN_TRAINING=True`, `ONLY_SEQUENCE='HoneyBee'`, `ONLY_VARIANT='rgbsplit_yc_a320'`, `MAX_RUNS=1` for the first pilot. Repeat with `ONLY_VARIANT='ycbcrsplit444_a320'`. Set `RUN_LATENCY=True` only after preparing `LATENCY_MANIFEST`, a JSON mapping all six architecture names in `benchmark_decoder_latency.py` to absolute checkpoint, config, command paths and SHA-256 hashes for each artifact. The selected archive contains only ten checkpoints; locate the remaining originals in the persistent archive rather than training replacements. Then set `ONLY_SEQUENCE=''`, `ONLY_VARIANT=''`, and `MAX_RUNS=0` and run the full 16-job campaign; supply Bunny inputs before the Bunny jobs. Re-running the launcher restores backups and resumes incomplete models.

`phase2_plan.csv`, `model_latest.pth`, `model_best.pth`, `epoch150.pth`, `epoch150.csv`, `completion.json`, `config.json`, `command.txt`, `environment.txt`, `git_commit.txt`, and logs are the expected training artifacts. Periodic backups go to `MyDrive/ChromaNeRV_CVPR_Phase2/runs`; latency CSVs and metadata go to the sibling `latency` directory. Export only CSV, JSON, logs, and the plan for paper writing; do not upload source frames or checkpoint weights to GitHub.

The original Bosphorus RGBSplit A320-W8 reference is seed 3, whereas the ChromaNeRV reference and new controls are seed 1. Keep both original checkpoints. Flag the comparison as mixed-seed and do not label it seed-matched. The new models add only a single 1.5M point; no new BD-rate curve follows from these runs. YCbCrSplit444's full-resolution chroma receives direct supervision only at half resolution through the Y/C term; its RGB consistency term also constrains full-resolution chroma.

Budget local disk for the two ZIPs plus extracted frames/checkpoints and temporary outputs. Inspect actual archive sizes in the notebook before the full campaign; dataset/checkpoint ZIP bytes and compression ratios depend on the user's Drive copies. A100 memory needs should be measured by a one-job pilot. The six-model latency result is a HoneyBee runtime comparison, not a multi-sequence runtime distribution; peak allocated memory is process-local.
