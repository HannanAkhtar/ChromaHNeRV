import argparse
import os

import numpy as np
import pandas as pd


RATE = "fixed_width_bpp"
RGB_QUALITY = "quant_rgb_psnr"
YUV_QUALITY = "quant_yuv_psnr_611_dbavg"
RD_METRICS = [
    RATE, RGB_QUALITY, "quant_psnr_y", "quant_psnr_cb", "quant_psnr_cr",
    YUV_QUALITY, "quant_rgb_ms_ssim", "quant_lpips_alex",
]


def family_name(row):
    experiment = str(row["experiment"])
    mode = str(row.get("branch_width_mode", "none"))
    if experiment == "rgb444_hnerv":
        return "full_rgb"
    if experiment == "ycbcr444_hnerv":
        return "full_ycbcr444"
    if experiment.startswith("rgbsplit_") and mode == "native":
        return "native_rgbsplit"
    if experiment.startswith("chroma420_") and mode == "native":
        return "native_chroma420"
    return f"{experiment}_{mode}"


def pareto_envelope(frame, quality):
    clean = frame.dropna(subset=[RATE, quality]).copy()
    clean = clean[(clean[RATE] > 0) & np.isfinite(clean[RATE]) & np.isfinite(clean[quality])]
    if clean.empty:
        return clean
    clean = clean.sort_values([RATE, quality], ascending=[True, False]).drop_duplicates(RATE, keep="first")
    keep, best = [], -np.inf
    for index, row in clean.iterrows():
        if row[quality] > best:
            keep.append(index)
            best = row[quality]
    return clean.loc[keep].sort_values(RATE)


def bd_pair(anchor, test, quality):
    anchor = pareto_envelope(anchor, quality)
    test = pareto_envelope(test, quality)
    result = {
        "valid": False, "reason": "", "anchor_points": len(anchor), "test_points": len(test),
        "overlap_rate_min": np.nan, "overlap_rate_max": np.nan,
        "overlap_quality_min": np.nan, "overlap_quality_max": np.nan,
        "bd_rate_percent": np.nan, "bd_psnr_db": np.nan, "curve_crossing": False,
        "interpolation": "monotonic_piecewise_linear",
    }
    if len(anchor) < 4 or len(test) < 4:
        result["reason"] = "fewer than four Pareto-valid RD points"
        return result

    rate_lo = max(anchor[RATE].min(), test[RATE].min())
    rate_hi = min(anchor[RATE].max(), test[RATE].max())
    quality_lo = max(anchor[quality].min(), test[quality].min())
    quality_hi = min(anchor[quality].max(), test[quality].max())
    result.update(overlap_rate_min=rate_lo, overlap_rate_max=rate_hi,
                  overlap_quality_min=quality_lo, overlap_quality_max=quality_hi)
    if rate_hi <= rate_lo or quality_hi <= quality_lo:
        result["reason"] = "curves have no common rate and quality interval"
        return result

    log_anchor = np.log(anchor[RATE].to_numpy())
    log_test = np.log(test[RATE].to_numpy())
    q_anchor = anchor[quality].to_numpy()
    q_test = test[quality].to_numpy()

    q_grid = np.linspace(quality_lo, quality_hi, 1000)
    anchor_log_rate = np.interp(q_grid, q_anchor, log_anchor)
    test_log_rate = np.interp(q_grid, q_test, log_test)
    mean_log_rate_delta = np.trapz(test_log_rate - anchor_log_rate, q_grid) / (quality_hi - quality_lo)
    result["bd_rate_percent"] = (np.exp(mean_log_rate_delta) - 1) * 100

    log_lo, log_hi = np.log(rate_lo), np.log(rate_hi)
    log_grid = np.linspace(log_lo, log_hi, 1000)
    anchor_quality = np.interp(log_grid, log_anchor, q_anchor)
    test_quality = np.interp(log_grid, log_test, q_test)
    quality_delta = test_quality - anchor_quality
    result["bd_psnr_db"] = np.trapz(quality_delta, log_grid) / (log_hi - log_lo)
    result["curve_crossing"] = bool(np.nanmin(quality_delta) < 0 < np.nanmax(quality_delta))
    result["valid"] = True
    return result


def curve_key(row):
    explicit = str(row.get("rd_curve", ""))
    if explicit and explicit.lower() != "nan":
        return explicit
    return "uniform" if row.get("quant_scheme") == "uniform" else "component_unassigned"


def build_rd_points(raw):
    frame = raw.copy()
    frame["model_family"] = frame.apply(family_name, axis=1)
    frame["curve"] = frame.apply(curve_key, axis=1)
    columns = [column for column in [
        "checkpoint_path", "run_name", "modelsize", "model_family", "experiment",
        "branch_width_mode", "split_stage", "curve", "quant_matrix", "quant_tag",
        "quant_configuration", *RD_METRICS,
    ] if column in frame.columns]
    return frame[columns].drop_duplicates(["checkpoint_path", "quant_configuration"], keep="last")


def comparisons_for_size(points):
    comparisons = []
    available = {(family, curve) for family, curve in zip(points["model_family"], points["curve"])}
    candidates = [
        ("full_rgb", "uniform", "native_chroma420", "uniform", "native Chroma uniform vs Full RGB uniform"),
        ("native_rgbsplit", "uniform", "native_chroma420", "uniform", "native Chroma vs converted native RGBSplit"),
        ("full_rgb", "uniform", "native_rgbsplit", "uniform", "converted native RGBSplit vs Full RGB"),
        ("full_ycbcr444", "uniform", "native_chroma420", "uniform", "native Chroma vs Full YCbCr444"),
    ]
    for curve in sorted(points.loc[points["model_family"] == "native_chroma420", "curve"].unique()):
        if curve not in ("uniform", "component_unassigned"):
            candidates.append(("full_rgb", "uniform", "native_chroma420", curve,
                               f"native Chroma {curve} vs Full RGB uniform"))
    for anchor_family, anchor_curve, test_family, test_curve, label in candidates:
        if (anchor_family, anchor_curve) in available and (test_family, test_curve) in available:
            comparisons.append((anchor_family, anchor_curve, test_family, test_curve, label))
    return comparisons


def build_bd_metrics(points):
    rows = []
    for modelsize, size_points in points.groupby("modelsize"):
        for anchor_family, anchor_curve, test_family, test_curve, label in comparisons_for_size(size_points):
            anchor = size_points[(size_points["model_family"] == anchor_family) & (size_points["curve"] == anchor_curve)]
            test = size_points[(size_points["model_family"] == test_family) & (size_points["curve"] == test_curve)]
            rgb = bd_pair(anchor, test, RGB_QUALITY)
            yuv = bd_pair(anchor, test, YUV_QUALITY)
            row = {
                "modelsize": modelsize, "comparison": label,
                "anchor_family": anchor_family, "anchor_curve": anchor_curve,
                "test_family": test_family, "test_curve": test_curve,
                "rgb_bd_rate_percent": rgb["bd_rate_percent"], "rgb_bd_psnr_db": rgb["bd_psnr_db"],
                "yuv_bd_rate_percent": yuv["bd_rate_percent"], "yuv_bd_psnr_db": yuv["bd_psnr_db"],
                "rgb_valid": rgb["valid"], "yuv_valid": yuv["valid"],
                "rgb_reason": rgb["reason"], "yuv_reason": yuv["reason"],
                "rgb_curve_crossing": rgb["curve_crossing"], "yuv_curve_crossing": yuv["curve_crossing"],
                "overlap_rate_min": rgb["overlap_rate_min"], "overlap_rate_max": rgb["overlap_rate_max"],
                "rgb_overlap_quality_min": rgb["overlap_quality_min"],
                "rgb_overlap_quality_max": rgb["overlap_quality_max"],
                "yuv_overlap_quality_min": yuv["overlap_quality_min"],
                "yuv_overlap_quality_max": yuv["overlap_quality_max"],
                "interpolation": rgb["interpolation"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build Stage 6 Pareto RD points and bounded BD metrics.")
    parser.add_argument("csv", help="Stage 6 raw PTQ CSV")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)
    raw = pd.read_csv(args.csv)
    points = build_rd_points(raw)
    points.to_csv(os.path.join(output_dir, "chroma_hnerv_stage6_rd_points.csv"), index=False)
    build_bd_metrics(points).to_csv(os.path.join(output_dir, "chroma_hnerv_stage6_bd_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
