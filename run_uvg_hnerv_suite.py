#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from uvg_utils import (
    UVG_FAMILIES, UVG_SEQUENCES, UVG_SIZES, UVG_WIDTHS,
    build_uvg_jobs, completion_status, load_uvg_manifest,
    restore_run_from_backup,
)


SMOKE_JOBS = (
    ("rgb444", 0.35, None),
    ("rgbsplit_a320", 0.35, 8),
    ("chroma420_a320", 0.35, 8),
    ("rgbsplit_a160", 3.0, 4),
    ("chroma420_a160", 3.0, 4),
)


def parse_args(argv=None):
    repo_default = str(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Sequential seven-sequence UVG HNeRV suite launcher.")
    parser.add_argument("--repo-root", default=repo_default)
    parser.add_argument("--data-root", default="data/uvg")
    parser.add_argument("--output-root", default="output/uvg7_hnerv_150e")
    parser.add_argument("--backup-root", default="")
    parser.add_argument("--manifest", default="configs/uvg7.json")
    parser.add_argument("--sequences", nargs="+", default=list(UVG_SEQUENCES))
    parser.add_argument("--sizes", nargs="+", type=float, default=list(UVG_SIZES))
    parser.add_argument("--widths", nargs="+", type=int, default=list(UVG_WIDTHS))
    parser.add_argument("--families", nargs="+", default=list(UVG_FAMILIES))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--strict-backup", action="store_true")
    return parser.parse_args(argv)


def resolve_launcher_paths(args):
    args.repo_root = str(Path(args.repo_root).resolve())
    args.data_root = str(Path(args.data_root).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    args.manifest = str(
        (Path(args.repo_root) / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest).resolve()
    )
    if args.backup_root:
        args.backup_root = str(Path(args.backup_root).resolve())
    return args


def selected_jobs(args):
    if args.smoke:
        all_jobs = build_uvg_jobs(
            sequences=["HoneyBee"], sizes=[0.35, 3.0],
            widths=[8, 4], families=UVG_FAMILIES, seed=args.seed)
        selected = []
        for family, size, width in SMOKE_JOBS:
            selected.append(next(
                job for job in all_jobs
                if job["family"] == family
                and job["modelsize"] == size
                and job["branch_width"] == width))
        return selected
    return build_uvg_jobs(
        sequences=args.sequences, sizes=args.sizes, widths=args.widths,
        families=args.families, seed=args.seed)


def build_command(args, job, metadata, run_dir, log_path):
    epochs = 1 if args.smoke else args.epochs
    command = [
        sys.executable, str(Path(args.repo_root) / "train_chroma_hnerv.py"),
        "--dataset_preset", "uvg_hnerv",
        "--data_path", str(Path(args.data_root) / job["sequence"]),
        "--vid", job["sequence"],
        "--run_name", job["run_id"],
        "--run_dir", str(run_dir),
        "--experiment", job["experiment"],
        "--modelsize", str(job["modelsize"]),
        "--manualSeed", str(args.seed),
        "--epochs", str(epochs),
        "--quant_model_bit", "8",
        "--quant_embed_bit", "6",
        "--intermediate_eval_mode", "quick",
        "--final_eval_mode", "full",
        "--expected_frames", str(metadata["frames"]),
        "--expected_source_height", str(metadata["source_height"]),
        "--expected_source_width", str(metadata["source_width"]),
        "--launcher_log_path", str(log_path),
    ]
    if job["split_stage"]:
        command += [
            "--split_stage", job["split_stage"],
            "--branch_width_mode", "fixed",
            "--branch_width", str(job["branch_width"]),
        ]
    if args.smoke:
        command += ["--eval_freq", "1", "--debug"]
    if args.backup_root:
        command += ["--backup_root", args.backup_root, "--backup_at_eval"]
        if args.strict_backup:
            command.append("--strict_backup")
    return command


def stream_job(command, log_path, gpu, repo_root, resume=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(shlex.join(command), flush=True)
    mode = "a" if resume else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if resume:
            separator = (
                f"\n{'=' * 24} RESUME "
                f"{datetime.now().astimezone().isoformat()} {'=' * 24}\n")
            print(separator, end="", flush=True)
            log.write(separator)
            log.flush()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment, cwd=str(repo_root))
        if process.stdout is None:
            raise RuntimeError("Could not capture trainer output.")
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def archive_launcher_log(log_path):
    if not log_path.is_file():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archived = log_path.with_name(f"{log_path.name}.previous.{timestamp}")
    counter = 1
    while archived.exists():
        archived = log_path.with_name(
            f"{log_path.name}.previous.{timestamp}.{counter}")
        counter += 1
    log_path.replace(archived)
    print(f"  ARCHIVED launcher log: {archived}", flush=True)
    return archived


def write_failure(run_dir, run_id, return_code=None, reason=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    failure = {"status": "failed", "run_id": run_id}
    if return_code is not None:
        failure["return_code"] = return_code
    if reason:
        failure["reason"] = reason
    with (run_dir / "failure.json").open("w", encoding="utf-8") as file:
        json.dump(failure, file, indent=2)


def require_complete_run(run_dir, epochs):
    if completion_status(run_dir, epochs) != "complete":
        raise RuntimeError(
            "trainer exited successfully without complete final artifacts")


def update_completion_restoration(run_dir, restored_from_backup):
    completion_path = Path(run_dir) / "completion.json"
    with completion_path.open(encoding="utf-8") as file:
        completion = json.load(file)
    completion["restored_from_backup"] = bool(
        completion.get("restored_from_backup", False) or restored_from_backup)
    temporary = completion_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(completion, file, indent=2, sort_keys=True)
    temporary.replace(completion_path)


def cleanup_stale_failure_markers(local_run_dir, backup_run_dir=None):
    removed = []
    directories = [Path(local_run_dir)]
    if backup_run_dir is not None:
        directories.append(Path(backup_run_dir))
    for directory in directories:
        marker = directory / "failure.json"
        if marker.is_file():
            try:
                marker.unlink()
                removed.append(str(marker))
                print(f"  REMOVED stale failure marker: {marker}", flush=True)
            except OSError as exc:
                print(
                    f"  WARNING could not remove stale failure marker "
                    f"{marker}: {exc}", flush=True)
    return removed


def sync_completion_to_backup(run_dir, backup_run_dir, strict=False):
    if backup_run_dir is None:
        return
    try:
        backup_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            Path(run_dir) / "completion.json",
            backup_run_dir / "completion.json")
    except OSError as exc:
        message = f"Could not update backup completion.json: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        print(f"  WARNING {message}", flush=True)


def validate_smoke_result(run_dir, job, metadata):
    csv_path = run_dir / "epoch1.csv"
    if completion_status(run_dir, 1) != "complete":
        raise RuntimeError(f"Smoke run did not produce complete artifacts: {run_dir}")
    frame = pd.read_csv(csv_path)
    if len(frame) != 1:
        raise RuntimeError(f"Expected one smoke result row in {csv_path}.")
    row = frame.iloc[0]
    finite_columns = [
        "rgb_psnr", "quant_rgb_psnr", "quant_psnr_y",
        "quant_psnr_cb", "quant_psnr_cr",
    ]
    invalid = [column for column in finite_columns if not math.isfinite(float(row[column]))]
    if invalid:
        raise RuntimeError(f"Non-finite smoke metrics in {csv_path}: {invalid}")
    expected = {
        "detected_frames": metadata["frames"],
        "detected_source_height": metadata["source_height"],
        "detected_source_width": metadata["source_width"],
        "final_output_height": 960,
        "final_output_width": 1920,
        "quant_model_bit": 8,
        "quant_embed_bit": 6,
    }
    if job["split_stage"] == "a320":
        expected.update(shared_output_height=480, shared_output_width=960)
    elif job["split_stage"] == "a160":
        expected.update(shared_output_height=160, shared_output_width=320)
    if job["experiment"].startswith("chroma420_"):
        expected.update(chroma_output_height=480, chroma_output_width=960)
    for field, value in expected.items():
        if int(row[field]) != value:
            raise RuntimeError(f"Smoke validation failed: {field}={row[field]}, expected {value}.")
    if job["branch_width"] is not None and int(row["branch_width"]) != job["branch_width"]:
        raise RuntimeError("Smoke branch width does not match the requested width.")


def main(argv=None):
    args = resolve_launcher_paths(parse_args(argv))
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    backup_root = Path(args.backup_root) if args.backup_root else None
    manifest = load_uvg_manifest(args.manifest)
    jobs = selected_jobs(args)
    if args.force and not args.yes:
        raise SystemExit("--force requires --yes because completed run directories may be deleted.")

    if args.smoke:
        output_root = output_root.parent / f"{output_root.name}_smoke"
    print(f"Planned UVG jobs: {len(jobs)}", flush=True)
    for index, job in enumerate(jobs, 1):
        run_dir = output_root / job["run_id"]
        backup_run_dir = backup_root / job["run_id"] if backup_root else None
        log_path = output_root / "logs" / f"{job['run_id']}.log"
        restoration = {
            "restored": [], "kept_local": [],
            "invalid_local": [], "invalid_backup": [],
        }
        if backup_run_dir is not None and not args.force and not args.dry_run:
            restoration = restore_run_from_backup(
                run_dir, backup_run_dir, 1 if args.smoke else args.epochs)
            if any(restoration.values()):
                print(f"  RESTORE {json.dumps(restoration, sort_keys=True)}", flush=True)
        state = completion_status(run_dir, 1 if args.smoke else args.epochs)
        print(
            f"[{index}/{len(jobs)}] {job['run_id']} state={state} "
            f"data={data_root / job['sequence']}", flush=True)
        if state == "complete" and not args.force:
            update_completion_restoration(run_dir, bool(restoration["restored"]))
            cleanup_stale_failure_markers(run_dir, backup_run_dir)
            sync_completion_to_backup(
                run_dir, backup_run_dir, args.strict_backup)
            print("  SKIP complete", flush=True)
            continue
        if args.force and not args.dry_run:
            archive_launcher_log(log_path)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            state = "new"
        if state == "resume":
            print("  RESUME from model_latest.pth", flush=True)

        sequence_dir = data_root / job["sequence"]
        if not args.dry_run and not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"Expected direct sequence frame directory: {sequence_dir}. "
                "Do not pass the parent UVG directory to the trainer.")
        command = build_command(
            args, job, manifest[job["sequence"]], run_dir, log_path)
        if args.dry_run:
            print("  " + shlex.join(command))
            continue

        return_code = stream_job(
            command, log_path, args.gpu, repo_root, resume=(state == "resume"))
        if return_code:
            write_failure(run_dir, job["run_id"], return_code=return_code)
            if not args.continue_on_error:
                raise SystemExit(return_code)
            continue
        try:
            require_complete_run(
                run_dir, 1 if args.smoke else args.epochs)
            if args.smoke:
                validate_smoke_result(
                    run_dir, job, manifest[job["sequence"]])
            update_completion_restoration(
                run_dir, bool(restoration["restored"]))
            cleanup_stale_failure_markers(run_dir, backup_run_dir)
            sync_completion_to_backup(
                run_dir, backup_run_dir, args.strict_backup)
        except Exception as exc:
            write_failure(run_dir, job["run_id"], reason=str(exc))
            if not args.continue_on_error:
                raise
            print(f"  FAILED validation: {exc}", flush=True)

    print(f"Suite phase complete. Planned job count: {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
