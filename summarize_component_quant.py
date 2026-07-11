import argparse
import os

import numpy as np
import pandas as pd


QUALITY_COLUMNS = [
    "quant_rgb_psnr", "quant_psnr_y", "quant_yuv_psnr_611_mse",
    "quant_lpips_alex", "fixed_width_bpp",
]


def _configuration(frame):
    if "quant_configuration" in frame:
        return frame["quant_configuration"].astype(str)
    return frame.get("quant_tag", pd.Series("", index=frame.index)).astype(str)


def _metric_delta(left, right, category):
    row = {
        "comparison": category,
        "left_checkpoint": left.get("checkpoint_path", ""),
        "left_quantization": left["quant_configuration"],
        "right_checkpoint": right.get("checkpoint_path", ""),
        "right_quantization": right["quant_configuration"],
    }
    for column in QUALITY_COLUMNS:
        row[f"{column}_difference"] = left.get(column, np.nan) - right.get(column, np.nan)
    return row


def build_pairwise(summary):
    rows = []
    for _, group in summary.groupby("checkpoint_path", dropna=False):
        uniform = group[group["quant_scheme"] == "uniform"]
        mixed = group[group["quant_scheme"] == "component"]
        for _, candidate in mixed.iterrows():
            if uniform.empty:
                continue
            baseline = uniform.iloc[(uniform["fixed_width_bpp"] - candidate["fixed_width_bpp"]).abs().argmin()]
            family = "Chroma" if str(candidate["experiment"]).startswith("chroma420_") else "RGBSplit"
            rows.append(_metric_delta(candidate, baseline, f"same {family} checkpoint: mixed minus uniform"))

    for _, chroma in summary[summary["experiment"].astype(str).str.startswith("chroma420_")].iterrows():
        rgb = summary[
            summary["experiment"].astype(str).str.startswith("rgbsplit_")
            & (summary["modelsize"] == chroma["modelsize"])
            & (summary["split_stage"] == chroma["split_stage"])
            & (summary["quant_shared_bit"] == chroma["quant_shared_bit"])
            & (summary["quant_rgb_bit"] == chroma["quant_y_bit"])
        ]
        # A literal same-bitwidth comparison exists when Chroma Y and C use the same width.
        if chroma.get("quant_y_bit", -1) != chroma.get("quant_chroma_bit", -1):
            rgb = rgb.iloc[0:0]
        if not rgb.empty:
            rows.append(_metric_delta(chroma, rgb.iloc[0], "same bitwidth: ChromaSplit minus RGBSplit"))

    full = summary[summary["experiment"] == "rgb444_hnerv"]
    chroma_mixed = summary[
        summary["experiment"].astype(str).str.startswith("chroma420_")
        & (summary["quant_scheme"] == "component")
    ]
    for _, full_row in full.iterrows():
        candidates = chroma_mixed[chroma_mixed["modelsize"] == full_row["modelsize"]]
        if not candidates.empty:
            nearest = candidates.iloc[(candidates["fixed_width_bpp"] - full_row["fixed_width_bpp"]).abs().argmin()]
            rows.append(_metric_delta(full_row, nearest, "full RGB uniform minus Chroma mixed"))
    return pd.DataFrame(rows)


def build_matched_bpp(summary):
    rows = []
    full = summary[(summary["experiment"] == "rgb444_hnerv") & (summary["quant_scheme"] == "uniform")]
    chroma = summary[
        summary["experiment"].astype(str).str.startswith("chroma420_")
        & (summary["quant_scheme"] == "component")
    ]
    for _, source in full.iterrows():
        candidates = chroma[chroma["modelsize"] == source["modelsize"]]
        if candidates.empty:
            continue
        target = candidates.iloc[(candidates["fixed_width_bpp"] - source["fixed_width_bpp"]).abs().argmin()]
        difference = target["fixed_width_bpp"] - source["fixed_width_bpp"]
        rows.append({
            "full_model": source.get("checkpoint_path", ""),
            "full_quantization_configuration": source["quant_configuration"],
            "chroma_model": target.get("checkpoint_path", ""),
            "chroma_quantization_configuration": target["quant_configuration"],
            "bpp_difference": difference,
            "relative_bpp_mismatch": abs(difference) / max(abs(source["fixed_width_bpp"]), 1e-12),
            "rgb_psnr_difference": target.get("quant_rgb_psnr", np.nan) - source.get("quant_rgb_psnr", np.nan),
            "y_psnr_difference": target.get("quant_psnr_y", np.nan) - source.get("quant_psnr_y", np.nan),
            "yuv_psnr_difference": target.get("quant_yuv_psnr_611_mse", np.nan) - source.get("quant_yuv_psnr_611_mse", np.nan),
            "lpips_difference": target.get("quant_lpips_alex", np.nan) - source.get("quant_lpips_alex", np.nan),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize component-aware ChromaHNeRV PTQ results.")
    parser.add_argument("csv", help="Raw consolidated quantization CSV")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    raw = pd.read_csv(args.csv)
    raw["quant_configuration"] = _configuration(raw)
    keys = ["checkpoint_path", "quant_configuration"]
    summary = raw.sort_index().drop_duplicates(keys, keep="last")
    summary.to_csv(os.path.join(output_dir, "chroma_quant_eval_summary.csv"), index=False)
    build_pairwise(summary).to_csv(os.path.join(output_dir, "chroma_quant_pairwise_deltas.csv"), index=False)
    build_matched_bpp(summary).to_csv(os.path.join(output_dir, "chroma_quant_matched_bpp.csv"), index=False)


if __name__ == "__main__":
    main()
