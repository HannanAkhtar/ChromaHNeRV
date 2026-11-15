#!/usr/bin/env python3
"""Strict 132-frame Phase 2 ablation launcher; never schedules reference models."""
import argparse
import csv
import json
import shlex
from pathlib import Path

from run_uvg_hnerv_suite import (cleanup_stale_failure_markers, require_complete_run,
                                 restore_run_from_backup, stream_job, write_failure)
from uvg_utils import discover_image_files

SEQUENCES = ("Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySetGo",
             "ShakeNDry", "YachtRide", "Bunny")
VARIANTS = ("rgbsplit_yc_a320", "ycbcrsplit444_a320")
REPO = Path(__file__).resolve().parent


def jobs():
    return [dict(sequence=sequence, variant=variant,
                 run_id=f"cvpr27_phase2_{sequence}_{variant}_1p5m_a320_w8_seed1")
            for sequence in SEQUENCES for variant in VARIANTS]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvg-root", default="data/uvg")
    parser.add_argument("--bunny-root", default="")
    parser.add_argument("--output-root", default="output/cvpr27_phase2")
    parser.add_argument("--backup-root", default="")
    parser.add_argument("--bunny-command", default="", help="Original Bunny command.txt; required for Bunny training")
    parser.add_argument("--manifest", default=str(REPO / "configs/cvpr27_phase2_132.json"))
    parser.add_argument("--only-sequence", choices=SEQUENCES)
    parser.add_argument("--only-variant", choices=VARIANTS)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--strict-backup", action="store_true")
    return parser.parse_args(argv)


def command_for(job, args, run_dir, source):
    is_bunny = job["sequence"] == "Bunny"
    command = ["python", str(REPO / "train_chroma_hnerv.py"),
               "--data_path", str(source), "--vid", job["sequence"],
               "--run_name", job["run_id"], "--run_dir", str(run_dir),
               "--experiment", job["variant"], "--split_stage", "a320",
               "--branch_width_mode", "fixed", "--branch_width", "8",
               "--modelsize", "1.5", "--manualSeed", "1", "--epochs", str(args.epochs),
               "--expected_frames", "132", "--quant_model_bit", "8",
               "--quant_embed_bit", "6", "--intermediate_eval_mode", "quick",
               "--final_eval_mode", "full", "--lambda_y", "1", "--lambda_c", "1",
               "--lambda_rgb", "0.1"]
    if is_bunny:
        if not args.bunny_command:
            raise ValueError("Bunny requires --bunny-command from the original checkpoint")
        historical = shlex.split(Path(args.bunny_command).read_text(encoding="utf-8"))
        for option in ("--enc_strds", "--dec_strds", "--crop_list", "--resize_list",
                       "--enc_dim", "--ks", "--reduce", "--lower_width", "--loss"):
            if option not in historical:
                raise ValueError(f"Original Bunny command lacks {option}")
            index = historical.index(option) + 1
            count = 5 if option in ("--enc_strds", "--dec_strds") else 1
            values = historical[index:index + count]
            if len(values) != count:
                raise ValueError(f"Original Bunny command has incomplete {option}")
            command.extend([option, *values])
    else:
        command.extend(["--dataset_preset", "uvg_hnerv", "--expected_source_height", "1080",
                        "--expected_source_width", "1920"])
    if args.backup_root:
        command.extend(["--backup_root", str(Path(args.backup_root).resolve()),
                        "--backup_every_epochs", "5", "--backup_at_eval"])
        if args.strict_backup:
            command.append("--strict_backup")
    return command


def preflight(job, source, metadata):
    files = discover_image_files(source)
    if len(files) != 132 or metadata["frames"] != 132:
        raise ValueError(f"{job['sequence']}: expected exactly 132 frames, found {len(files)}")
    from PIL import Image
    sizes = {Image.open(path).size for path in files}
    if len(sizes) != 1:
        raise ValueError(f"{job['sequence']}: inconsistent frame dimensions: {sizes}")
    width, height = sizes.pop()
    if height < metadata["crop_height"] or width < metadata["crop_width"]:
        raise ValueError(f"{job['sequence']}: source smaller than crop")
    if "source_height" in metadata and (height, width) != (metadata["source_height"], metadata["source_width"]):
        raise ValueError(f"{job['sequence']}: expected source {metadata['source_height']}x{metadata['source_width']}, got {height}x{width}")


def main(argv=None):
    args = parse_args(argv)
    metadata = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if set(metadata) != set(SEQUENCES):
        raise ValueError("Phase 2 manifest must contain exactly eight sequences")
    selected = [j for j in jobs() if (not args.only_sequence or j["sequence"] == args.only_sequence)
                and (not args.only_variant or j["variant"] == args.only_variant)]
    if args.max_runs < 0:
        raise ValueError("--max-runs cannot be negative")
    if args.max_runs:
        selected = selected[:args.max_runs]
    output = Path(args.output_root).resolve()
    backup = Path(args.backup_root).resolve() if args.backup_root else None
    plan = []
    for job in selected:
        source = (Path(args.bunny_root) if job["sequence"] == "Bunny" else
                  Path(args.uvg_root) / job["sequence"]).resolve()
        command = command_for(job, args, output / job["run_id"], source) if (job["sequence"] != "Bunny" or args.bunny_command) else []
        plan.append({**job, "source": str(source), "run_dir": str(output / job["run_id"]),
                     "command": shlex.join(command) if command else "REQUIRES --bunny-command"})
    print(f"Planned Phase 2 jobs: {len(selected)}")
    for item in plan:
        print(json.dumps(item))
    if args.dry_run or args.list:
        return plan
    for job, item in zip(selected, plan):
        if job["sequence"] == "Bunny" and not args.bunny_root:
            raise FileNotFoundError("Bunny archive must be extracted and --bunny-root supplied")
        if job["sequence"] == "Bunny" and not args.bunny_command:
            raise FileNotFoundError("Provide original Bunny --bunny-command for historical preprocessing")
        preflight(job, Path(item["source"]), metadata[job["sequence"]])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "phase2_plan.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=plan[0].keys() if plan else
                                ("sequence", "variant", "run_id", "source", "run_dir", "command"))
        writer.writeheader(); writer.writerows(plan)
    for job, item in zip(selected, plan):
        run_dir = Path(item["run_dir"])
        backup_run = backup / job["run_id"] if backup else None
        if backup_run:
            print("RESTORE", restore_run_from_backup(run_dir, backup_run, args.epochs))
        try:
            require_complete_run(run_dir, args.epochs)
            print("SKIP complete", job["run_id"])
            continue
        except RuntimeError:
            pass
        if run_dir.exists() and not (run_dir / "model_latest.pth").exists():
            raise RuntimeError(f"Incomplete run without resumable checkpoint: {run_dir}")
        log = output / "logs" / (job["run_id"] + ".log")
        command = command_for(job, args, run_dir, Path(item["source"]))
        command.extend(["--launcher_log_path", str(log)])
        command[0] = __import__("sys").executable
        code = stream_job(command, log, args.gpu, REPO, resume=(run_dir / "model_latest.pth").exists())
        if code:
            write_failure(run_dir, job["run_id"], return_code=code)
            raise RuntimeError(f"Trainer failed for {job['run_id']}: {code}")
        require_complete_run(run_dir, args.epochs)
        cleanup_stale_failure_markers(run_dir, backup_run)
    return plan


if __name__ == "__main__":
    main()
