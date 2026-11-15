# CVPR 2027 Phase 2 runbook

## Inputs and environment

Open `notebooks/ChromaNeRV_CVPR_Phase2_Training_and_Latency.ipynb` from branch
`experiment/cvpr27-phase2-controls-latency` in a Colab A100 runtime. The notebook
clones and fast-forwards that branch, prints its commit SHA, checks the installed
Python/PyTorch/torchvision/CUDA/GPU stack, and installs only missing or incompatible
non-PyTorch requirements. It validates imports, runs the complete test suite, and
records versions in `MyDrive/ChromaNeRV_CVPR_Phase2/phase2_environment.json` and
`phase2_environment.txt`. A broken torch/torchvision CUDA pair is a hard error;
do not downgrade a working A100 stack to the historical pinned versions.

Place `ChromaNeRV_UVG132.zip` and
`ChromaNeRV_CVPR_Checkpoints_20261115_021417.zip` somewhere under MyDrive.
Optionally set `UVG_EXPECTED_SHA256` and `CHECKPOINT_EXPECTED_SHA256` to hashes
from a **trusted prior audit**. The notebook streams SHA-256, checks ZIP structure
and member CRCs, rejects path traversal/symlinks, validates already-extracted
files by size and CRC, and repairs only missing or corrupt members. It records
archive sizes and hashes in `phase2_archive_audit.json`. A hash calculated from
the supplied ZIP alone does not establish original provenance. UVG preflight
checks all seven sequences for 132 original 1080x1920 frames; training uses the
960x1920 crop. Bunny must come from its own original 132-frame ZIP and original
`command.txt`, with 640x1280 crop and historical preprocessing. Do not substitute
UVG inputs/settings for Bunny.

## Execute

1. **Preflight:** Leave `RUN_TRAINING=False` and `RUN_LATENCY=False`, then run all
   notebook cells. Confirm archive audit, test-suite pass, seven UVG preflights,
   and `Planned Phase 2 jobs: 16`. Set `BUNNY_ZIP` and `BUNNY_COMMAND` before Bunny
   work. The configuration defaults to one HoneyBee RGBSplit + Y/C run.
2. **Pilot 1:** Set `RUN_TRAINING=True`, `ONLY_SEQUENCE='HoneyBee'`,
   `ONLY_VARIANT='rgbsplit_yc_a320'`, `MAX_RUNS=1`, `EPOCHS=150`. Run the notebook
   training cell. Confirm `epoch150.pth`, `epoch150.csv`, `quant_vid.pth`,
   `completion.json`, and persistent Drive backup.
3. **Pilot 2:** Change only `ONLY_VARIANT='ycbcrsplit444_a320'` and run the training
   cell again. Confirm the same artifacts. These are the first two of the 16
   campaign jobs, not extra duplicates; completed jobs are skipped on rerun.
4. **Latency:** On an otherwise idle A100, set `RUN_LATENCY=True` and run the
   latency cell. It discovers the six original HoneyBee epoch-150 checkpoints,
   configs and commands recursively from the extracted checkpoint archive. If
   the selected ZIP lacks any, set `LATENCY_EXTRA_ROOT` to an extracted original
   archive containing them, or supply a complete `LATENCY_MANIFEST`. The benchmark
   fails on missing artifacts; never train replacements. Results and the generated
   SHA-256 manifest are saved under `MyDrive/ChromaNeRV_CVPR_Phase2/latency`.
5. **Campaign:** Set `RUN_LATENCY=False`, `ONLY_SEQUENCE=''`,
   `ONLY_VARIANT=''`, `MAX_RUNS=0`, `EPOCHS=150`, with `BUNNY_ZIP` and
   `BUNNY_COMMAND` populated. Run the training cell. Exactly 16 distinct
   sequence/variant jobs are planned; completed pilots are skipped. The original
   recovered checkpoints are read only.

The launcher backs up `model_latest.pth`, optimizer state, metadata, logs and
available CSVs after **every epoch** with atomic replacement on Drive, then
backs up final artifacts. After a Colab disconnect, rerun the notebook cells;
the launcher restores valid backups, keeps a higher-epoch local checkpoint,
resumes optimizer/epoch state, and validates quantized final CSV plus checkpoint
before marking a job complete. Do not delete the persistent `runs` directory.

## Metrics and interpretation

`estimated_gflops` remains the historical full-forward Conv/Linear estimate,
which includes encoder operations. `decoder_gflops` is the new playback estimate:
the encoder embedding is computed before profiler hooks are registered, then the
decoder is called with `input_embed`. Both use two FLOPs per multiply-accumulate
for Conv2d, ConvTranspose2d and Linear. Activations, normalization, interpolation,
colour conversion, reshaping, memory traffic and kernel overhead are omitted.
The two estimates must not be substituted for each other. `decoder_params_M` and
`decoder_playback_params` report actual architecture counts, not assumed equality.

The latency benchmark precomputes all 132 HoneyBee frame embeddings and checks
cached-versus-ordinary decoding on first, middle and last frames. It uses batch 1,
FP32, rotating cached embeddings, 50 warm-ups, five
trials of 200 decodes by default. CUDA events are preallocated per trial and
synchronized once per trial; repeated-trial timings, mean, median, standard
deviation, p5, p95, FPS, and speedup against Full RGB are saved. The process-local
peak allocated delta excludes resident model and embedding. GPU temperature,
clocks, other processes and sequential model order can affect timing; telemetry
and process checks are recorded. Run the six models on the same otherwise idle A100.

Use `merge_cvpr27_phase2.py --new-root <runs> --references <original.csv>
--output <merged>` for per-sequence and aggregate results. It never changes
original CSVs. Bosphorus RGBSplit reference is seed 3; ChromaNeRV and the new
controls are seed 1. Mixed-seed Bosphorus deltas remain visible and flagged in
`phase2_ablation_pairs.csv` and the inclusive summary, while
`phase2_ablation_seed_matched_summary.csv` excludes them. FP32 and M8/E6 rows
are separate, missing originals or quantized metrics appear in
`phase2_missing_sources.csv`, and no new BD-rate is inferred from a single point.
