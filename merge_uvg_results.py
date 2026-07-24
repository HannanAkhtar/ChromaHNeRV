#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from uvg_utils import (
    UVG_FAMILIES, UVG_SEQUENCES, UVG_SIZES, UVG_WIDTHS, build_uvg_jobs,
)


KEY_COLUMNS = [
    "vid", "experiment", "split_stage", "branch_width", "branch_width_mode",
    "modelsize", "manualSeed", "quant_model_bit", "quant_embed_bit",
]


def expected_rows(seed=1):
    return build_uvg_jobs(
        UVG_SEQUENCES, UVG_SIZES, UVG_WIDTHS, UVG_FAMILIES, seed)


def normalized_key(row):
    split = str(row.get("split_stage", "none"))
    if split.lower() == "nan":
        split = "none"
    width = row.get("branch_width")
    width = None if pd.isna(width) else int(width)
    mode = str(row.get("branch_width_mode", "none"))
    return (
        str(row["vid"]), str(row["experiment"]), split, width, mode,
        float(row["modelsize"]), int(row["manualSeed"]),
        int(row["quant_model_bit"]), int(row["quant_embed_bit"]),
    )


def expected_key(job):
    return (
        job["sequence"], job["experiment"], job["split_stage"] or "none",
        job["branch_width"], job["branch_width_mode"],
        job["modelsize"], job["manualSeed"], 8, 6,
    )


def merge_results(output_root, destination, seed=1):
    output_root, destination = Path(output_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    recorded_failure_dirs = set()
    for failure_path in output_root.rglob("failure.json"):
        try:
            details = json.loads(failure_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            details = {"reason": f"invalid failure.json: {exc}"}
        failures.append({
            "run_dir": str(failure_path.parent),
            "csv_path": "",
            "reason": details.get("reason", f"trainer return code {details.get('return_code', 'unknown')}"),
        })
        recorded_failure_dirs.add(failure_path.parent)
    for csv_path in output_root.rglob("epoch*.csv"):
        run_dir = csv_path.parent
        completion = run_dir / "completion.json"
        try:
            status = json.loads(completion.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "csv_path": str(csv_path), "reason": str(exc)})
            continue
        if status != "complete" and run_dir not in recorded_failure_dirs:
            failures.append({
                "run_dir": str(run_dir), "csv_path": str(csv_path),
                "reason": "completion.json is absent or not complete",
            })
            continue
        for _, row in frame.iterrows():
            record = row.to_dict()
            record["result_file"] = str(csv_path)
            if int(record.get("quant_model_bit", -1)) != 8 or int(record.get("quant_embed_bit", -1)) != 6:
                failures.append({
                    "run_dir": str(run_dir), "csv_path": str(csv_path),
                    "reason": "final result is not M8/E6",
                })
                continue
            record["reported_variant"] = "quant_m8_e6"
            records.append(record)

    merged = pd.DataFrame(records)
    by_key = {}
    if not merged.empty:
        for index, row in merged.iterrows():
            by_key.setdefault(normalized_key(row), []).append(index)

    duplicate_rows = []
    for key, indices in by_key.items():
        if len(indices) > 1:
            duplicate_rows.append({
                **dict(zip(KEY_COLUMNS, key)),
                "count": len(indices),
                "conflicting_file_paths": "|".join(merged.loc[indices, "result_file"].astype(str)),
            })

    expected = {expected_key(job): job for job in expected_rows(seed)}
    missing_rows = [
        {**job, "reason": "no unique complete M8/E6 result"}
        for key, job in expected.items()
        if key not in by_key or len(by_key[key]) != 1
    ]
    unique_indices = [indices[0] for indices in by_key.values() if len(indices) == 1]
    merged_unique = merged.loc[unique_indices].copy() if unique_indices else pd.DataFrame(columns=merged.columns)
    sort_columns = [
        column for column in ["vid", "modelsize", "experiment", "split_stage", "branch_width"]
        if column in merged_unique
    ]
    if sort_columns:
        merged_unique = merged_unique.sort_values(sort_columns, na_position="first")

    merged_unique.to_csv(destination / "uvg7_all_runs.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(destination / "uvg7_missing_runs.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(destination / "uvg7_duplicate_runs.csv", index=False)
    pd.DataFrame(failures).to_csv(destination / "uvg7_failed_runs.csv", index=False)
    return merged_unique, missing_rows, duplicate_rows, failures


def main():
    parser = argparse.ArgumentParser(description="Merge per-run UVG7 HNeRV results.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", default="")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    destination = args.destination or args.output_root
    merged, missing, duplicates, failures = merge_results(
        args.output_root, destination, args.seed)
    print(
        f"Merged={len(merged)} missing={len(missing)} "
        f"duplicates={len(duplicates)} failures={len(failures)}")


if __name__ == "__main__":
    main()
