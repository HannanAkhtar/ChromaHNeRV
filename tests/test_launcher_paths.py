from pathlib import Path

from run_uvg_hnerv_suite import (
    archive_launcher_log, build_command, parse_args, resolve_launcher_paths,
    stream_job,
)


def test_relative_launcher_paths_become_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = resolve_launcher_paths(parse_args([
        "--repo-root", "repo", "--data-root", "data",
        "--output-root", "output", "--manifest", "configs/uvg7.json",
    ]))
    assert all(Path(value).is_absolute() for value in [
        args.repo_root, args.data_root, args.output_root, args.manifest])
    assert args.backup_root == ""


def test_strict_backup_is_forwarded(tmp_path):
    args = resolve_launcher_paths(parse_args([
        "--repo-root", str(tmp_path), "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "output"),
        "--backup-root", str(tmp_path / "backup"), "--strict-backup",
    ]))
    job = {
        "sequence": "Beauty", "run_id": "run", "experiment": "rgb444_hnerv",
        "modelsize": 0.35, "split_stage": None, "branch_width": None,
    }
    command = build_command(
        args, job, {"frames": 600, "source_height": 1080, "source_width": 1920},
        tmp_path / "run", tmp_path / "run.log")
    assert "--strict_backup" in command
    assert Path(command[1]).is_absolute()


def test_stream_job_sets_repo_cwd(tmp_path, monkeypatch):
    captured = {}

    class Process:
        stdout = []

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(
        "run_uvg_hnerv_suite.subprocess.Popen", fake_popen)
    assert stream_job(
        ["python", "trainer.py"], tmp_path / "run.log", "0", tmp_path) == 0
    assert captured["cwd"] == str(tmp_path)


def test_forced_run_archives_old_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("old run", encoding="utf-8")
    archived = archive_launcher_log(log)
    assert archived.read_text(encoding="utf-8") == "old run"
    assert not log.exists()
