import json

from uvg_utils import completion_status


def test_completion_requires_all_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "epoch150.csv").write_text("x\n1\n")
    assert completion_status(run, 150) == "new"
    (run / "model_latest.pth").write_bytes(b"latest")
    assert completion_status(run, 150) == "resume"
    (run / "epoch150.pth").write_bytes(b"epoch")
    (run / "model_best.pth").write_bytes(b"best")
    (run / "completion.json").write_text(json.dumps({"status": "complete"}))
    assert completion_status(run, 150) == "complete"
