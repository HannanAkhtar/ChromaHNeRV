#!/usr/bin/env python3
"""Merge Phase 2 single-budget measurements; never infer a new BD-rate curve."""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_cvpr27_phase2 import SEQUENCES

VARIANTS = ("rgbsplit_a320", "rgbsplit_yc_a320", "ycbcrsplit444_a320", "chroma420_a320")
NEW_VARIANTS = VARIANTS[1:3]
QUALITY = ("rgb_psnr", "psnr_y", "yuv_psnr_611_dbavg", "yuv_ssim_611", "lpips_alex", "vmaf")
STRUCTURE = ("decoder_gflops", "estimated_gflops", "decoder_params_M", "actual_total_params_M")


def _row(raw, sequence, variant, precision, source):
    prefix = "quant_" if precision == "m8e6" else ""
    row = {"vid": sequence, "experiment": variant, "precision": precision,
           "source": str(source), "manualSeed": raw.get("manualSeed"),
           "epoch": raw.get("cur_epoch", raw.get("epochs")),
           "modelsize": raw.get("modelsize"), "loss": raw.get("loss"),
           "objective": raw.get("objective", raw.get("loss")),
           "lambda_y": raw.get("lambda_y"), "lambda_c": raw.get("lambda_c"),
           "lambda_rgb": raw.get("lambda_rgb"),
           "quant_model_bit": raw.get("quant_model_bit"),
           "quant_embed_bit": raw.get("quant_embed_bit"),
           "decoder_playback_params": raw.get("decoder_playback_params"),
           "git_commit": raw.get("git_commit")}
    for metric in QUALITY:
        row[metric] = raw.get(prefix + metric, float("nan"))
    for metric in STRUCTURE:
        row[metric] = raw.get(metric, float("nan"))
    return row


def _valid_number(value):
    return pd.notna(value) and str(value).strip() != ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--references", required=True, help="Original per-sequence reference CSV; never edited")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    reference = pd.read_csv(args.references)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for sequence in SEQUENCES:
        for variant in VARIANTS:
            if variant in NEW_VARIANTS:
                run = Path(args.new_root) / f"cvpr27_phase2_{sequence}_{variant}_1p5m_a320_w8_seed1"
                completion, path = run / "completion.json", run / "epoch150.csv"
                if not completion.is_file() or not path.is_file():
                    missing.append({"vid": sequence, "experiment": variant, "reason": "missing final artifacts"})
                    continue
                status = json.loads(completion.read_text(encoding="utf-8"))
                if status.get("status") != "complete" or status.get("epoch") != 150:
                    missing.append({"vid": sequence, "experiment": variant, "reason": "incomplete run"})
                    continue
                frame = pd.read_csv(path)
            else:
                path = Path(args.references)
                frame = reference[(reference.vid == sequence) & (reference.experiment == variant)]
            if len(frame) != 1:
                missing.append({"vid": sequence, "experiment": variant,
                                "reason": f"expected one source row; found {len(frame)}"})
                continue
            raw = frame.iloc[0]
            if raw.get("experiment") != variant or raw.get("vid") != sequence:
                raise ValueError(f"Unexpected source identity in {path}: {sequence}/{variant}")
            if not _valid_number(raw.get("manualSeed")) or not _valid_number(raw.get("modelsize")):
                raise ValueError(f"Source lacks seed or budget: {path}")
            if float(raw.get("modelsize")) != 1.5 or float(raw.get("cur_epoch", raw.get("epochs", -1))) != 150:
                raise ValueError(f"Expected epoch 150, nominal 1.5M source: {path}")
            rows.append(_row(raw, sequence, variant, "fp32", path))
            if _valid_number(raw.get("quant_rgb_psnr")):
                if (int(raw.get("quant_model_bit", -1)) != 8 or
                        int(raw.get("quant_embed_bit", -1)) != 6):
                    raise ValueError(f"Quantized source is not M8/E6: {path}")
                rows.append(_row(raw, sequence, variant, "m8e6", path))
            else:
                missing.append({"vid": sequence, "experiment": variant, "reason": "missing M8/E6 metrics"})
    result = pd.DataFrame(rows, columns=list(_row({}, "", "", "fp32", "").keys()))
    result.to_csv(output / "phase2_ablation_per_sequence.csv", index=False)
    pd.DataFrame(missing, columns=["vid", "experiment", "reason"]).to_csv(
        output / "phase2_missing_sources.csv", index=False)
    paired = []
    for sequence in SEQUENCES:
        for precision in ("fp32", "m8e6"):
            subset = result[(result.vid == sequence) & (result.precision == precision)]
            base = subset[subset.experiment == "rgbsplit_a320"]
            if base.empty:
                continue
            original = base.iloc[0]
            for _, row in subset[subset.experiment != "rgbsplit_a320"].iterrows():
                seed_matched = int(row.manualSeed) == int(original.manualSeed)
                protocol_matched = (row.epoch == original.epoch and row.modelsize == original.modelsize and
                                    (precision == "fp32" or
                                     (row.quant_model_bit == original.quant_model_bit and
                                      row.quant_embed_bit == original.quant_embed_bit)))
                item = {"vid": sequence, "comparison": row.experiment, "precision": precision,
                        "reference_seed": original.manualSeed, "variant_seed": row.manualSeed,
                        "seed_matched": seed_matched, "mixed_seed": not seed_matched,
                        "protocol_matched": protocol_matched,
                        "reference_source": original.source, "variant_source": row.source}
                for metric in QUALITY + STRUCTURE:
                    a, b = row[metric], original[metric]
                    item[f"delta_{metric}"] = (a - b if protocol_matched and
                                                _valid_number(a) and _valid_number(b) else float("nan"))
                paired.append(item)
    pair_columns = (["vid", "comparison", "precision", "reference_seed", "variant_seed",
                     "seed_matched", "mixed_seed", "protocol_matched", "reference_source",
                     "variant_source"] + [f"delta_{metric}" for metric in QUALITY + STRUCTURE])
    paired = pd.DataFrame(paired, columns=pair_columns)
    paired.to_csv(output / "phase2_ablation_pairs.csv", index=False)
    for filename, strict in (("phase2_ablation_summary.csv", False),
                             ("phase2_ablation_seed_matched_summary.csv", True)):
        subset = paired[paired.protocol_matched & paired.seed_matched] if strict else paired[paired.protocol_matched]
        summaries = []
        if not subset.empty:
            for (variant, precision), group in subset.groupby(["comparison", "precision"]):
                item = {"comparison": variant, "precision": precision,
                        "n_sequences": len(group), "n_mixed_seed": int(group.mixed_seed.sum()),
                        "included_sequences": ",".join(group.vid)}
                for metric in QUALITY + STRUCTURE:
                    values = group[f"delta_{metric}"].dropna()
                    item[f"mean_delta_{metric}"] = values.mean() if not values.empty else float("nan")
                    item[f"median_delta_{metric}"] = values.median() if not values.empty else float("nan")
                summaries.append(item)
        columns = (["comparison", "precision", "n_sequences", "n_mixed_seed", "included_sequences"] +
                   [f"{stat}_delta_{metric}" for metric in QUALITY + STRUCTURE
                    for stat in ("mean", "median")])
        pd.DataFrame(summaries, columns=columns).to_csv(output / filename, index=False)
    if missing:
        print(f"Missing source/metric records: {len(missing)}; see phase2_missing_sources.csv")
    if not paired.empty and paired.mixed_seed.any():
        print("Mixed-seed comparisons retained and flagged; use seed-matched summary for strict aggregate")


if __name__ == "__main__":
    main()
