import json

import pandas as pd

from merge_uvg_results import merge_results


def result_row(model_bit=8):
    return {
        "vid": "Beauty",
        "experiment": "rgb444_hnerv",
        "split_stage": "none",
        "branch_width": float("nan"),
        "branch_width_mode": "none",
        "modelsize": 0.35,
        "manualSeed": 1,
        "quant_model_bit": model_bit,
        "quant_embed_bit": 6,
        "rgb_psnr": 30.0,
        "quant_rgb_psnr": 29.8,
    }


def write_run(root, name, row):
    run = root / name
    run.mkdir()
    pd.DataFrame([row]).to_csv(run / "epoch150.csv", index=False)
    (run / "completion.json").write_text(json.dumps({"status": "complete"}))


def test_merger_reports_duplicates_missing_and_invalid_quantization(tmp_path):
    write_run(tmp_path, "first", result_row())
    write_run(tmp_path, "duplicate", result_row())
    write_run(tmp_path, "invalid", {
        **result_row(model_bit=7), "vid": "Bosphorus"})
    destination = tmp_path / "merged"
    merged, missing, duplicates, failures = merge_results(tmp_path, destination)
    assert merged.empty
    assert duplicates and duplicates[0]["count"] == 2
    assert missing
    assert any("not M8/E6" in failure["reason"] for failure in failures)
    duplicate_frame = pd.read_csv(destination / "uvg7_duplicate_runs.csv")
    assert "conflicting_file_paths" in duplicate_frame
