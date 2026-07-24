#!/usr/bin/env python3
"""Resumable Stage 6 native/full-width training, conversion, evaluation, and PTQ launcher."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_DEFAULT = "/content/ChromaHNeRV"
DRIVE_DEFAULT = "/content/drive/MyDrive/ChromaHNeRV_runs"
SIZES = (0.35, 0.75, 1.5, 3.0)
BITS = tuple(range(8, 1, -1))


def size_tag(size):
    return str(size).replace(".", "p")


def normalized_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    normalized = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("blocks.0."):
            name = name[len("blocks.0."):]
        normalized[name] = value
    return normalized


def validate_signature(path, family):
    checkpoint = torch.load(path, map_location="cpu")
    state = normalized_state(checkpoint)
    keys = set(state)
    if family in ("full_rgb", "full_ycbcr"):
        required, forbidden = ("decoder.", "head_layer."), ("shared_decoder.", "rgb_branch.", "y_branch.")
    elif family == "native_rgb":
        required, forbidden = ("shared_decoder.", "rgb_branch.", "rgb_head."), ("decoder.", "y_branch.")
    elif family == "native_chroma":
        required = ("shared_decoder.", "y_branch.", "y_head.", "cbcr_branch.", "cbcr_head.")
        forbidden = ("decoder.", "rgb_branch.", "rgb_head.")
    else:
        raise ValueError(f"Unknown family: {family}")
    missing = [prefix for prefix in required if not any(key.startswith(prefix) for key in keys)]
    bad = [prefix for prefix in forbidden if any(key.startswith(prefix) for key in keys)]
    if missing or bad:
        raise ValueError(f"Checkpoint signature mismatch for {family}: missing={missing}, forbidden={bad}: {path}")
    if family.startswith("native_"):
        mode = checkpoint.get("architecture_config", {}).get("branch_width_mode")
        conversion_mode = checkpoint.get("stage6_conversion", {}).get("branch_width_mode")
        if mode != "native" and conversion_mode != "native":
            raise ValueError(f"Checkpoint has split keys but no native-width architecture signature: {path}")
    return {"epoch": checkpoint.get("epoch"), "tensor_count": len(state)}


def locate_full_checkpoint(drive_root, family, size):
    """Locate an existing Stage 1-4 full HNeRV checkpoint robustly.

    Prefer the exact known run-folder naming used by the earlier stages, then
    fall back to a broader recursive search. Rejected candidates are reported
    instead of being silently discarded.
    """
    output_root = drive_root / "output"
    if not output_root.is_dir():
        raise FileNotFoundError(f"Drive output root does not exist: {output_root}")

    tag = size_tag(size)
    if family == "full_rgb":
        exact_run_tokens = [
            f"rgb444_hnerv_rgb_hnerv_{tag}m_bunny",
            f"rgb444_hnerv_rgb444_hnerv_{tag}m_bunny",
        ]
        family_tokens = ("rgb444_hnerv", "rgb_hnerv")
        forbidden_path_tokens = ("ycbcr",)
    elif family == "full_ycbcr":
        exact_run_tokens = [
            f"ycbcr444_hnerv_ycbcr_hnerv_{tag}m_bunny",
            f"ycbcr444_hnerv_ycbcr444_hnerv_{tag}m_bunny",
        ]
        family_tokens = ("ycbcr444_hnerv", "ycbcr_hnerv")
        forbidden_path_tokens = ()
    else:
        raise ValueError(f"Unsupported full-model family: {family}")

    all_epoch150 = sorted(output_root.rglob("epoch150.pth"))
    preferred = [
        path for path in all_epoch150
        if any(token in str(path).lower() for token in exact_run_tokens)
        and "stage6_" not in str(path).lower()
    ]

    # Broader fallback for slightly different historical folder names.
    size_tokens = (
        f"{tag}m",
        f"size{size}",
        f"size{float(size)}",
    )
    fallback = [
        path for path in all_epoch150
        if "stage6_" not in str(path).lower()
        and any(token in str(path).lower() for token in family_tokens)
        and any(token in str(path).lower() for token in size_tokens)
        and not any(token in str(path).lower() for token in forbidden_path_tokens)
    ]

    candidates = []
    for path in preferred + fallback:
        if path not in candidates:
            candidates.append(path)

    accepted = []
    rejected = []
    for path in candidates:
        try:
            info = validate_signature(path, family)
            checkpoint = torch.load(path, map_location="cpu")
            recorded_size = checkpoint.get("architecture_config", {}).get("modelsize")
            if recorded_size is not None and abs(float(recorded_size) - float(size)) > 1e-9:
                rejected.append((path, f"recorded modelsize={recorded_size}"))
                continue
            accepted.append((path, info))
        except Exception as error:
            rejected.append((path, f"{type(error).__name__}: {error}"))

    if not accepted:
        diagnostic = [
            f"No Stage 1-4 {family} epoch150 checkpoint found for {size}M.",
            f"Scanned output root: {output_root}",
            f"Total epoch150 checkpoints seen: {len(all_epoch150)}",
            f"Path candidates after name filtering: {len(candidates)}",
        ]
        if rejected:
            diagnostic.append("Rejected candidates:")
            diagnostic.extend(f"  - {path}: {reason}" for path, reason in rejected)
        else:
            diagnostic.append("No path matched the expected family/size tokens.")
            diagnostic.append(f"Expected run-folder tokens: {exact_run_tokens}")
        raise FileNotFoundError("\n".join(diagnostic))

    # Prefer Stage 1 for 0.35/0.75/1.5 and Stage 4 for 3.0, then shortest path.
    def preference(item):
        path = item[0]
        lowered = str(path).lower()
        preferred_stage_penalty = (
            0 if ((size < 3.0 and "stage1_" in lowered) or
                  (size >= 3.0 and "stage4_" in lowered))
            else 1
        )
        return (preferred_stage_penalty, len(str(path)), str(path))

    accepted.sort(key=preference)
    selected, info = accepted[0]
    print(f"Selected {family} {size}M: {selected}")
    print(f"  epoch={info.get('epoch')} tensors={info.get('tensor_count')}")
    if len(accepted) > 1:
        print("  Other valid candidates:")
        for path, _ in accepted[1:]:
            print(f"    - {path}")
    return selected


def converted_path(train_root, size):
    return train_root / f"rgbsplit_a160_native_{size_tag(size)}m" / "epoch150.pth"


def locate_native_chroma(train_root, size):
    token = f"chroma420_a160_native_{size_tag(size)}m"
    candidates = sorted(path for path in train_root.rglob("epoch150.pth") if token in str(path))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one native Chroma checkpoint containing '{token}', found {len(candidates)}.")
    validate_signature(candidates[0], "native_chroma")
    return candidates[0]


def common_model_args(repo_root, output_root, family, size, run_name):
    experiment = {
        "full_rgb": "rgb444_hnerv", "full_ycbcr": "ycbcr444_hnerv",
        "native_rgb": "rgbsplit_a160", "native_chroma": "chroma420_a160",
    }[family]
    command = [
        sys.executable, str(repo_root / "train_chroma_hnerv.py"),
        "--data_path", "data/bunny", "--vid", "bunny", "--outf", str(output_root),
        "--run_name", run_name, "--experiment", experiment, "--modelsize", str(size),
        "-e", "150", "-b", "2", "--eval_freq", "30", "--workers", "2",
        "--conv_type", "convnext", "pshuffel", "--act", "gelu", "--norm", "none",
        "--crop_list", "640_1280", "--resize_list", "-1", "--loss", "L2",
        "--enc_strds", "5", "4", "4", "2", "2", "--enc_dim", "64_16",
        "--dec_strds", "5", "4", "4", "2", "2", "--ks", "0_1_5",
        "--reduce", "1.2", "--lower_width", "12", "--lr", "0.001", "--manualSeed", "1",
    ]
    if family.startswith("native_"):
        command += ["--split_stage", "a160", "--branch_width_mode", "native"]
    if family == "native_chroma":
        command += ["--lambda_y", "1", "--lambda_c", "1", "--lambda_rgb", "0.1",
                    "--chroma_scale", "2", "--chroma_upsample", "bilinear"]
    return command


def run_command(command, log_path, dry_run=False):
    print(shlex.join([str(item) for item in command]))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def completed_ptq(csv_path):
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path)
    if not {"checkpoint_path", "quant_tag"}.issubset(frame):
        return set()
    return {(str(row.checkpoint_path), str(row.quant_tag)) for row in frame.itertuples()}


def completed_runs(csv_path):
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path)
    return set(frame["run_name"].dropna().astype(str)) if "run_name" in frame else set()


def quant_configs(family, phase):
    uniform = [{"tag": f"U{bit}E6", "matrix": "uniform", "curve": "uniform",
                "args": ["--quant_scheme", "uniform", "--quant_model_bit", str(bit)]} for bit in BITS]
    if phase == "ptq_uniform":
        return uniform
    configs = []
    if family == "native_chroma":
        patterns = [
            ("chroma_only", lambda b: (8, 8, b)),
            ("shared_only", lambda b: (b, 8, 8)),
            ("luma_only", lambda b: (8, b, 8)),
            ("shared_luma_c8", lambda b: (b, b, 8)),
        ]
        for matrix, values in patterns:
            for bit in BITS:
                shared, y_bit, chroma = values(bit)
                configs.append({
                    "tag": f"S{shared}Y{y_bit}C{chroma}E6", "matrix": matrix,
                    "curve": f"chroma_{matrix}",
                    "args": ["--quant_scheme", "component", "--quant_shared_bit", str(shared),
                             "--quant_y_bit", str(y_bit), "--quant_chroma_bit", str(chroma)],
                })
    elif family == "native_rgb":
        patterns = [
            ("shared_only", lambda b: (b, 8)),
            ("rgb_tail_only", lambda b: (8, b)),
            ("shared_rgb", lambda b: (b, b)),
        ]
        for matrix, values in patterns:
            for bit in BITS:
                shared, rgb = values(bit)
                configs.append({
                    "tag": f"S{shared}RGB{rgb}E6", "matrix": matrix, "curve": f"rgb_{matrix}",
                    "args": ["--quant_scheme", "component", "--quant_shared_bit", str(shared),
                             "--quant_rgb_bit", str(rgb)],
                })
    unique = {}
    for config in configs:
        unique.setdefault(config["tag"], config)
    return list(unique.values())


def smoke_configs():
    return [
        ("full_rgb", "U8E6", ["--quant_scheme", "uniform", "--quant_model_bit", "8"]),
        ("full_rgb", "U5E6", ["--quant_scheme", "uniform", "--quant_model_bit", "5"]),
        ("native_rgb", "U8E6", ["--quant_scheme", "uniform", "--quant_model_bit", "8"]),
        ("native_rgb", "U5E6", ["--quant_scheme", "uniform", "--quant_model_bit", "5"]),
        ("native_chroma", "U8E6", ["--quant_scheme", "uniform", "--quant_model_bit", "8"]),
        ("native_chroma", "U5E6", ["--quant_scheme", "uniform", "--quant_model_bit", "5"]),
        ("native_chroma", "S6Y6C8E6", ["--quant_scheme", "component", "--quant_shared_bit", "6", "--quant_y_bit", "6", "--quant_chroma_bit", "8"]),
        ("native_chroma", "S5Y5C8E6", ["--quant_scheme", "component", "--quant_shared_bit", "5", "--quant_y_bit", "5", "--quant_chroma_bit", "8"]),
        ("native_chroma", "S8Y8C5E6", ["--quant_scheme", "component", "--quant_shared_bit", "8", "--quant_y_bit", "8", "--quant_chroma_bit", "5"]),
        ("native_chroma", "S8Y8C7E6", ["--quant_scheme", "component", "--quant_shared_bit", "8", "--quant_y_bit", "8", "--quant_chroma_bit", "7"]),
    ]


def model_checkpoint(drive_root, train_root, family, size):
    if family in ("full_rgb", "full_ycbcr"):
        return locate_full_checkpoint(drive_root, family, size)
    if family == "native_rgb":
        path = converted_path(train_root, size)
        validate_signature(path, family)
        return path
    return locate_native_chroma(train_root, size)


def phase_inspect(drive_root, train_root):
    for size in SIZES:
        for family in ("full_rgb", "full_ycbcr"):
            locate_full_checkpoint(drive_root, family, size)
        for family, path in (("native_rgb", converted_path(train_root, size)),):
            print(f"Stage 6 {family} {size}M: {'ready' if path.exists() else 'missing'} {path}")
        try:
            print(f"Stage 6 native_chroma {size}M: {locate_native_chroma(train_root, size)}")
        except RuntimeError as error:
            print(error)


def phase_profile(args, repo, results, train_root, logs):
    raw = results / "chroma_hnerv_stage6_architecture_profile_raw.csv"
    done = set() if args.force else completed_runs(raw)
    for size in SIZES:
        for family in ("full_rgb", "full_ycbcr", "native_rgb", "native_chroma"):
            key = f"{family}_{size_tag(size)}m"
            run_name = f"stage6_profile_{key}"
            if run_name in done:
                print(f"SKIP completed profile: {run_name}")
                continue
            command = common_model_args(repo, train_root / "profile", family, size, run_name)
            command += ["--profile_only", "--results_csv", str(raw), "--quant_model_bit", "-1", "--quant_embed_bit", "-1"]
            run_command(command, logs / "profile" / f"{key}.log", args.dry_run)
    if args.dry_run or not raw.exists():
        return
    frame = pd.read_csv(raw).drop_duplicates(["run_name"], keep="last")
    frame["native_rgb_param_match_full"] = False
    frame["native_rgb_gflops_match_full"] = False
    for size in SIZES:
        full = frame[(frame.modelsize == size) & (frame.experiment == "rgb444_hnerv")]
        split = frame[(frame.modelsize == size) & (frame.experiment == "rgbsplit_a160")]
        if not full.empty and not split.empty:
            split_index = split.index[-1]
            frame.loc[split_index, "native_rgb_param_match_full"] = (
                int(split.iloc[-1].actual_total_params) == int(full.iloc[-1].actual_total_params))
            frame.loc[split_index, "native_rgb_gflops_match_full"] = (
                abs(split.iloc[-1].estimated_gflops - full.iloc[-1].estimated_gflops)
                <= max(1e-6, abs(full.iloc[-1].estimated_gflops) * 1e-4))
    frame.to_csv(results / "chroma_hnerv_stage6_architecture_manifest.csv", index=False)


def phase_convert(args, repo, drive, train_root, logs):
    utility = repo / "convert_full_hnerv_to_native_rgbsplit.py"
    for size in SIZES:
        source, output = locate_full_checkpoint(drive, "full_rgb", size), converted_path(train_root, size)
        if output.exists() and not args.force:
            validate_signature(output, "native_rgb")
            print(f"SKIP converted control: {output}")
            continue
        command = [sys.executable, str(utility), "--checkpoint", str(source), "--output", str(output),
                   "--modelsize", str(size), "--split_stage", "a160", "--verify"]
        if args.force:
            command.append("--force")
        run_command(command, logs / "convert" / f"rgb_{size_tag(size)}m.log", args.dry_run)


def phase_train(args, repo, train_root, results, logs):
    raw = results / "chroma_hnerv_stage6_train_raw.csv"
    families = ["native_chroma"] + (["native_rgb"] if args.train_rgb_controls else [])
    for size in SIZES:
        for family in families:
            key = f"{family}_{size_tag(size)}m"
            existing = list(train_root.rglob("epoch150.pth"))
            completion_token = (f"chroma420_a160_native_{size_tag(size)}m" if family == "native_chroma"
                                else f"rgbsplit_a160_native_{size_tag(size)}m")
            completed_paths = [path for path in existing if completion_token in str(path)]
            if family == "native_rgb":
                completed_paths = [path for path in completed_paths
                                   if "stage6_conversion" not in torch.load(path, map_location="cpu")]
            if completed_paths and not args.force:
                print(f"SKIP completed training: {key}")
                continue
            run_name = f"{'chroma420_a160' if family == 'native_chroma' else 'rgbsplit_a160'}_native_{size_tag(size)}m"
            command = common_model_args(repo, train_root, family, size, run_name)
            command += ["--results_csv", str(raw), "--quant_model_bit", "-1", "--quant_embed_bit", "-1"]
            run_command(command, logs / "train" / f"{key}.log", args.dry_run)


def phase_eval(args, repo, drive, train_root, eval_root, results, logs):
    raw = results / "chroma_hnerv_stage6_eval_raw.csv"
    done = set() if args.force else completed_runs(raw)
    for size in SIZES:
        for family in ("full_rgb", "full_ycbcr", "native_rgb", "native_chroma"):
            checkpoint = model_checkpoint(drive, train_root, family, size)
            key = f"{family}_{size_tag(size)}m"
            run_name = f"stage6_eval_{key}"
            if run_name in done:
                print(f"SKIP completed evaluation: {run_name}")
                continue
            command = common_model_args(repo, eval_root, family, size, run_name)
            command += ["--eval_only", "--not_resume", "--weight", str(checkpoint),
                        "--results_csv", str(raw), "--quant_model_bit", "-1", "--quant_embed_bit", "-1"]
            run_command(command, logs / "eval" / f"{key}.log", args.dry_run)


def phase_ptq(args, phase, repo, drive, train_root, ptq_root, results, logs):
    raw = results / "chroma_hnerv_stage6_ptq_raw.csv"
    done = set() if args.force else completed_ptq(raw)
    jobs = []
    if phase == "ptq_smoke":
        for family, tag, quant_args in smoke_configs():
            jobs.append((family, 1.5, {"tag": tag, "matrix": "smoke", "curve": "uniform" if tag.startswith("U") else "smoke_component", "args": quant_args}))
    else:
        families = ("full_rgb", "full_ycbcr", "native_rgb", "native_chroma") if phase == "ptq_uniform" else ("native_rgb", "native_chroma")
        for size in SIZES:
            for family in families:
                for config in quant_configs(family, phase):
                    jobs.append((family, size, config))
    for index, (family, size, config) in enumerate(jobs, 1):
        checkpoint = model_checkpoint(drive, train_root, family, size)
        key = (str(checkpoint), config["tag"])
        print(f"[{index}/{len(jobs)}] {family} {size}M {config['tag']} checkpoint={checkpoint}")
        if key in done:
            print("SKIP completed stable key")
            continue
        run_name = f"stage6_ptq_{family}_{size_tag(size)}m_{config['tag']}"
        command = common_model_args(repo, ptq_root, family, size, run_name)
        command += ["--eval_only", "--not_resume", "--weight", str(checkpoint), "--results_csv", str(raw),
                    "--quant_embed_bit", "6", "--quant_axis", "0", "--quant_storage_mode", "packed",
                    "--save_packed_quant", "--quant_tag", config["tag"],
                    "--quant_matrix", config["matrix"], "--rd_curve", config["curve"], *config["args"]]
        run_command(command, logs / phase / f"{family}_{size_tag(size)}m_{config['tag']}.log", args.dry_run)
        done.add(key)


def phase_summarize(args, repo, results, logs):
    raw = results / "chroma_hnerv_stage6_ptq_raw.csv"
    if not raw.exists() and not args.dry_run:
        raise FileNotFoundError(raw)
    temporary = results / "stage6_summary_tmp"
    if not args.dry_run:
        temporary.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, str(repo / "summarize_component_quant.py"), str(raw), "--output_dir", str(temporary)],
        [sys.executable, str(repo / "summarize_stage6_rd.py"), str(raw), "--output_dir", str(results)],
    ]
    for index, command in enumerate(commands):
        run_command(command, logs / "summarize" / f"summary_{index}.log", args.dry_run)
    if args.dry_run:
        return
    renames = {
        "chroma_quant_eval_summary.csv": "chroma_hnerv_stage6_ptq_summary.csv",
        "chroma_quant_pairwise_deltas.csv": "chroma_hnerv_stage6_pairwise_deltas.csv",
    }
    for source, target in renames.items():
        source_path = temporary / source
        if source_path.exists():
            source_path.replace(results / target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=[
        "inspect", "profile", "convert_rgb_controls", "train", "eval", "ptq_smoke",
        "ptq_uniform", "ptq_component", "summarize", "all"])
    parser.add_argument("--repo_root", default=REPO_DEFAULT)
    parser.add_argument("--drive_root", default=DRIVE_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--train_rgb_controls", action="store_true")
    args = parser.parse_args()

    repo, drive = Path(args.repo_root), Path(args.drive_root)
    train_root = drive / "output" / "stage6_fullwidth_a160_150e"
    eval_root = drive / "output" / "stage6_fullwidth_a160_150e_eval"
    ptq_root = drive / "output" / "stage6_fullwidth_ptq"
    results, logs = drive / "results", drive / "output" / "stage6_fullwidth_logs"
    for path in (train_root, eval_root, ptq_root, results, logs):
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)
    if not repo.is_dir():
        raise FileNotFoundError(repo)

    phases = (["inspect", "profile", "convert_rgb_controls", "train", "eval", "ptq_smoke",
               "ptq_uniform", "ptq_component", "summarize"] if args.phase == "all" else [args.phase])
    for phase in phases:
        print(f"\n{'=' * 100}\nStage 6 phase: {phase}\n{'=' * 100}")
        if phase == "inspect":
            phase_inspect(drive, train_root)
        elif phase == "profile":
            phase_profile(args, repo, results, train_root, logs)
        elif phase == "convert_rgb_controls":
            phase_convert(args, repo, drive, train_root, logs)
        elif phase == "train":
            phase_train(args, repo, train_root, results, logs)
        elif phase == "eval":
            phase_eval(args, repo, drive, train_root, eval_root, results, logs)
        elif phase.startswith("ptq_"):
            phase_ptq(args, phase, repo, drive, train_root, ptq_root, results, logs)
        elif phase == "summarize":
            phase_summarize(args, repo, results, logs)


if __name__ == "__main__":
    main()
