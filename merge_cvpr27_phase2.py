#!/usr/bin/env python3
"""Merge single-budget ablations without conflating FP32 and PTQ evaluations."""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_cvpr27_phase2 import SEQUENCES

VARIANTS = ("rgbsplit_a320", "rgbsplit_yc_a320", "ycbcrsplit444_a320", "chroma420_a320")
METRICS = ("rgb_psnr", "psnr_y", "yuv_psnr_611_dbavg", "yuv_ssim_611",
           "lpips_alex", "estimated_gflops", "actual_total_params_M", "vmaf")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--references", required=True, help="CSV containing original per-sequence reference rows")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    reference = pd.read_csv(args.references)
    rows = []
    for sequence in SEQUENCES:
        for variant in VARIANTS:
            if variant in ("rgbsplit_yc_a320", "ycbcrsplit444_a320"):
                run = Path(args.new_root) / f"cvpr27_phase2_{sequence}_{variant}_1p5m_a320_w8_seed1"
                completion = run / "completion.json"
                path = run / "epoch150.csv"
                if not completion.exists() or not path.exists():
                    continue
                if json.loads(completion.read_text(encoding="utf-8")).get("status") != "complete":
                    continue
                frame = pd.read_csv(path)
            else:
                frame = reference[(reference.vid == sequence) & (reference.experiment == variant)]
            for _, raw in frame.iterrows():
                for precision, prefix in (("fp32", ""), ("m8e6", "quant_")):
                    if precision == "m8e6" and not any(f"quant_{metric}" in raw for metric in METRICS):
                        continue
                    row = {"vid": sequence, "experiment": variant, "precision": precision,
                       "source": str(path) if variant in ("rgbsplit_yc_a320", "ycbcrsplit444_a320") else args.references,
                       "manualSeed": raw.get("manualSeed", 1 if variant in ("rgbsplit_yc_a320", "ycbcrsplit444_a320") else float("nan")),
                       "epoch": raw.get("cur_epoch", raw.get("epochs", float("nan"))),
                       "quant_model_bit": raw.get("quant_model_bit", float("nan")),
                       "quant_embed_bit": raw.get("quant_embed_bit", float("nan")),
                       "code_sha": raw.get("git_commit", "missing")}
                    for metric in METRICS:
                        row[metric] = raw.get(prefix + metric, raw.get(metric, float("nan")) if metric in ("estimated_gflops", "actual_total_params_M", "vmaf") else float("nan"))
                    rows.append(row)
    result = pd.DataFrame(rows)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "phase2_ablation_per_sequence.csv", index=False)
    summary = []
    for sequence in SEQUENCES:
        subset = result[result.vid == sequence]
        base = subset[subset.experiment == "rgbsplit_a320"]
        if base.empty:
            continue
        for _, row in subset.iterrows():
            if row.experiment == "rgbsplit_a320":
                continue
            same_precision = base[base.precision == row.precision]
            if same_precision.empty:
                continue
            original = same_precision.iloc[0]
            matched = (row.manualSeed == original.manualSeed and
                       row.quant_model_bit == original.quant_model_bit and
                       row.quant_embed_bit == original.quant_embed_bit and
                       row.epoch == original.epoch)
            item = {"vid": sequence, "comparison": row.experiment, "precision": row.precision,
                    "matched": bool(matched), "reference_seed": original.manualSeed,
                    "variant_seed": row.manualSeed}
            for metric in METRICS:
                item[f"delta_{metric}"] = row[metric] - original[metric] if matched else float("nan")
            summary.append(item)
    paired = pd.DataFrame(summary)
    if not paired.empty:
        aggregates = paired.groupby(["comparison", "precision"], as_index=False).agg({
            **{f"delta_{metric}": ["mean", "median"] for metric in METRICS}})
        aggregates.columns = ["_".join(map(str, name)).strip("_") for name in aggregates.columns]
        aggregates.to_csv(output / "phase2_ablation_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["comparison", "matched"]).to_csv(
            output / "phase2_ablation_summary.csv", index=False)
    if not paired.empty and not paired.matched.all():
        print("WARNING: unmatched seeds/epochs/quantization; corresponding paired deltas are NaN")


if __name__ == "__main__":
    main()
