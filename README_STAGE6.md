# Stage 6: Native Decoder Splits and 2-8-bit PTQ

Stage 6 adds `--branch_width_mode native` for split models. Fixed mode remains the default and preserves all W4/W8 tensor shapes and checkpoint keys.

In native mode, RGBSplit reuses the Full HNeRV decoder channel schedule and divides the same decoder modules into `shared_decoder` and `rgb_branch`. A converted native RGBSplit is therefore structurally and numerically equivalent to its Full RGB source. Native Chroma420 uses the same full-width shared and Y stages plus a native half-resolution CbCr stage, so its additional chroma cost is reported rather than treated as parameter-matched.

## Accounting

- `decoder_playback_params`: transmitted non-encoder model parameters.
- `fixed_width_weight_bits`: sum of each quantization-group parameter count times its assigned precision.
- `fixed_width_total_bits`: packed weight and embedding payloads, per-tensor padding, FP16 min/scale values, buffers, and metadata.
- `fixed_width_bpp`: `fixed_width_total_bits / (frame pixels * frame count)` and the primary Stage 6 rate.
- Huffman fields are secondary entropy estimates and do not imply integer-kernel acceleration.

The architecture manifest records actual parameter counts, group percentages, GFLOPs, and native channel schedules. Native Chroma is not assumed to match the nominal Full HNeRV parameter target.

## Core Commands

Train a native Chroma model:

```bash
python train_chroma_hnerv.py --experiment chroma420_a160 --split_stage a160 \
  --branch_width_mode native --modelsize 1.5 [dataset and architecture arguments]
```

Convert a Full RGB checkpoint without retraining:

```bash
python convert_full_hnerv_to_native_rgbsplit.py \
  --checkpoint /path/to/full/epoch150.pth \
  --output /path/to/rgbsplit_a160_native/epoch150.pth \
  --modelsize 1.5 --split_stage a160 --verify
```

Evaluate 5-bit uniform PTQ:

```bash
python train_chroma_hnerv.py --eval_only --not_resume --weight /path/to/epoch150.pth \
  --experiment chroma420_a160 --split_stage a160 --branch_width_mode native \
  --quant_scheme uniform --quant_model_bit 5 --quant_embed_bit 6 \
  --quant_storage_mode packed --save_packed_quant [dataset and architecture arguments]
```

Run Stage 6 from Colab:

```bash
python run_fullwidth_stage6.py --phase inspect
python run_fullwidth_stage6.py --phase profile
python run_fullwidth_stage6.py --phase convert_rgb_controls
python run_fullwidth_stage6.py --phase train
python run_fullwidth_stage6.py --phase ptq_smoke
python run_fullwidth_stage6.py --phase all
```

Use `--dry_run` to print commands only, `--force` to rerun completed jobs, and `--train_rgb_controls` only for an optional from-scratch RGBSplit control.

## Outputs

Stage 6 writes its checkpoints below `output/stage6_fullwidth_*` and these result files under the Drive `results` directory:

- `chroma_hnerv_stage6_architecture_manifest.csv`
- `chroma_hnerv_stage6_train_raw.csv`
- `chroma_hnerv_stage6_eval_raw.csv`
- `chroma_hnerv_stage6_ptq_raw.csv`
- `chroma_hnerv_stage6_ptq_summary.csv`
- `chroma_hnerv_stage6_pairwise_deltas.csv`
- `chroma_hnerv_stage6_rd_points.csv`
- `chroma_hnerv_stage6_bd_metrics.csv`

BD calculations use Pareto-filtered points, require at least four overlapping points, do not extrapolate, and flag curve crossings. Negative BD-Rate favors the test Chroma curve; positive BD-PSNR means a quality gain.
