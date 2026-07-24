import json
import os
import pickle
import sys
import types

from uvg_utils import completion_status, restore_run_from_backup


def install_fake_torch(monkeypatch):
    def load(path, map_location=None):
        with open(path, "rb") as file:
            return pickle.load(file)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=load))


def checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump({"value": value}, file)


def complete_backup(path, epochs=150):
    checkpoint(path / "model_latest.pth", "latest")
    checkpoint(path / "model_best.pth", "best")
    checkpoint(path / f"epoch{epochs}.pth", "final")
    (path / f"epoch{epochs}.csv").write_text("rgb_psnr\n30\n", encoding="utf-8")
    (path / "completion.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8")


def test_absent_local_complete_backup_becomes_complete(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    backup.mkdir()
    complete_backup(backup)
    summary = restore_run_from_backup(local, backup, 150)
    assert summary["restored"]
    assert completion_status(local, 150) == "complete"


def test_absent_local_latest_only_becomes_resume(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    checkpoint(backup / "model_latest.pth", "latest")
    restore_run_from_backup(local, backup, 150)
    assert completion_status(local, 150) == "resume"


def test_newer_local_is_kept(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    checkpoint(backup / "model_latest.pth", "backup")
    checkpoint(local / "model_latest.pth", "local")
    os.utime(backup / "model_latest.pth", (10, 10))
    os.utime(local / "model_latest.pth", (20, 20))
    summary = restore_run_from_backup(local, backup, 150)
    assert "model_latest.pth" in summary["kept_local"]


def test_newer_backup_replaces_local(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    checkpoint(backup / "model_latest.pth", "backup")
    checkpoint(local / "model_latest.pth", "local")
    os.utime(local / "model_latest.pth", (10, 10))
    os.utime(backup / "model_latest.pth", (20, 20))
    restore_run_from_backup(local, backup, 150)
    assert pickle.loads((local / "model_latest.pth").read_bytes())["value"] == "backup"


def test_newer_corrupt_backup_does_not_replace_valid_local(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    checkpoint(local / "model_latest.pth", "local")
    backup.mkdir()
    (backup / "model_latest.pth").write_bytes(b"corrupt")
    os.utime(local / "model_latest.pth", (10, 10))
    os.utime(backup / "model_latest.pth", (20, 20))
    summary = restore_run_from_backup(local, backup, 150)
    assert "model_latest.pth" in summary["invalid_backup"]
    assert pickle.loads((local / "model_latest.pth").read_bytes())["value"] == "local"


def test_valid_backup_replaces_corrupt_local(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    checkpoint(backup / "model_latest.pth", "backup")
    local.mkdir()
    (local / "model_latest.pth").write_bytes(b"corrupt")
    summary = restore_run_from_backup(local, backup, 150)
    assert "model_latest.pth" in summary["invalid_local"]
    assert pickle.loads((local / "model_latest.pth").read_bytes())["value"] == "backup"


def test_malformed_completion_does_not_replace_valid_local(tmp_path, monkeypatch):
    install_fake_torch(monkeypatch)
    backup, local = tmp_path / "backup", tmp_path / "local"
    backup.mkdir()
    local.mkdir()
    (backup / "completion.json").write_text("{bad", encoding="utf-8")
    (local / "completion.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8")
    os.utime(local / "completion.json", (10, 10))
    os.utime(backup / "completion.json", (20, 20))
    summary = restore_run_from_backup(local, backup, 150)
    assert "completion.json" in summary["invalid_backup"]
    assert json.loads((local / "completion.json").read_text())["status"] == "complete"
