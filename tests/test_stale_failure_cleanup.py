import json

import pandas as pd

from merge_uvg_results import merge_results
from run_uvg_hnerv_suite import cleanup_stale_failure_markers


def write_complete_run(run):
    run.mkdir()
    row = {
        "vid": "Beauty", "experiment": "rgb444_hnerv",
        "split_stage": "none", "branch_width": float("nan"),
        "branch_width_mode": "none", "modelsize": 0.35,
        "manualSeed": 1, "quant_model_bit": 8, "quant_embed_bit": 6,
    }
    pd.DataFrame([row]).to_csv(run / "epoch150.csv", index=False)
    (run / "epoch150.pth").write_bytes(b"final")
    (run / "model_best.pth").write_bytes(b"best")
    (run / "completion.json").write_text(
        json.dumps({"status": "complete", "epoch": 150}), encoding="utf-8")


def test_failure_marker_removed_after_success(tmp_path):
    local, backup = tmp_path / "local", tmp_path / "backup"
    local.mkdir()
    backup.mkdir()
    (local / "failure.json").write_text("{}")
    (backup / "failure.json").write_text("{}")
    removed = cleanup_stale_failure_markers(local, backup)
    assert len(removed) == 2
    assert not (local / "failure.json").exists()
    assert not (backup / "failure.json").exists()


def test_stale_failure_does_not_report_complete_run_failed(tmp_path):
    run = tmp_path / "complete"
    write_complete_run(run)
    (run / "failure.json").write_text(json.dumps({"status": "failed"}))
    _, _, _, failures = merge_results(tmp_path, tmp_path / "merged")
    assert not any(item["run_dir"] == str(run) for item in failures)


def test_incomplete_failed_run_is_reported(tmp_path):
    run = tmp_path / "failed"
    run.mkdir()
    (run / "failure.json").write_text(
        json.dumps({"status": "failed", "reason": "interrupted"}))
    _, _, _, failures = merge_results(tmp_path, tmp_path / "merged")
    assert any(item["run_dir"] == str(run) for item in failures)
