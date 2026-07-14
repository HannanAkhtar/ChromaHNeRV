#!/usr/bin/env python3
"""
Stage 5 component-aware PTQ sweep for ChromaHNeRV.

Repository:
    /content/ChromaHNeRV

Google Drive root:
    /content/drive/MyDrive/ChromaHNeRV_runs

This launcher never trains. Every job uses:
    --eval_only --not_resume --weight <epoch150.pth>

Phases:
    inspect  : locate and validate all required checkpoints only
    smoke    : 5 representative evaluations
    stage5a  : exhaustive 1.5M characterization (19 total configurations)
    c3       : optional 3-bit chroma boundary at 1.5M (3 configurations)
    stage5b  : replication at 0.75M and two 3.0M operating points (54 configurations)
    all      : stage5a + stage5b models with the standard matrices
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch


DEFAULT_REPO_ROOT = "/content/ChromaHNeRV"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ChromaHNeRV_runs"
DEFAULT_STAGE5_NAME = "stage5_component_ptq"


MODEL_SPECS: List[Dict] = [
    # Full RGB baselines.
    {
        "model_id": "full_rgb_0p75",
        "source_stage": "stage1_rgb_vs_ycbcr_150e",
        "source_run_dir": "rgb444_hnerv_rgb_hnerv_0p75m_bunny",
        "experiment": "rgb444_hnerv",
        "modelsize": "0.75",
        "split_stage": None,
        "branch_width": None,
    },
    {
        "model_id": "full_rgb_1p5",
        "source_stage": "stage1_rgb_vs_ycbcr_150e",
        "source_run_dir": "rgb444_hnerv_rgb_hnerv_1p5m_bunny",
        "experiment": "rgb444_hnerv",
        "modelsize": "1.5",
        "split_stage": None,
        "branch_width": None,
    },
    {
        "model_id": "full_rgb_3p0",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "rgb444_hnerv_rgb444_hnerv_3p0m_bunny",
        "experiment": "rgb444_hnerv",
        "modelsize": "3.0",
        "split_stage": None,
        "branch_width": None,
    },

    # 0.75M A160-W4.
    {
        "model_id": "rgb_a160_w4_0p75",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "rgbsplit_a160_rgbsplit_a160_w4_0p75m_bunny",
        "experiment": "rgbsplit_a160",
        "modelsize": "0.75",
        "split_stage": "a160",
        "branch_width": 4,
    },
    {
        "model_id": "chroma_a160_w4_0p75",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "chroma420_a160_chroma420_a160_w4_0p75m_bunny",
        "experiment": "chroma420_a160",
        "modelsize": "0.75",
        "split_stage": "a160",
        "branch_width": 4,
    },

    # 1.5M A160-W4.
    {
        "model_id": "rgb_a160_w4_1p5",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "rgbsplit_a160_rgbsplit_a160_w4_1p5m_bunny",
        "experiment": "rgbsplit_a160",
        "modelsize": "1.5",
        "split_stage": "a160",
        "branch_width": 4,
    },
    {
        "model_id": "chroma_a160_w4_1p5",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "chroma420_a160_chroma420_a160_w4_1p5m_bunny",
        "experiment": "chroma420_a160",
        "modelsize": "1.5",
        "split_stage": "a160",
        "branch_width": 4,
    },

    # 3.0M aggressive A160-W8.
    {
        "model_id": "rgb_a160_w8_3p0",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "rgbsplit_a160_rgbsplit_a160_w8_3p0m_bunny",
        "experiment": "rgbsplit_a160",
        "modelsize": "3.0",
        "split_stage": "a160",
        "branch_width": 8,
    },
    {
        "model_id": "chroma_a160_w8_3p0",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "chroma420_a160_chroma420_a160_w8_3p0m_bunny",
        "experiment": "chroma420_a160",
        "modelsize": "3.0",
        "split_stage": "a160",
        "branch_width": 8,
    },

    # 3.0M quality-efficient A320-W4.
    {
        "model_id": "rgb_a320_w4_3p0",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "rgbsplit_a320_rgbsplit_a320_w4_3p0m_bunny",
        "experiment": "rgbsplit_a320",
        "modelsize": "3.0",
        "split_stage": "a320",
        "branch_width": 4,
    },
    {
        "model_id": "chroma_a320_w4_3p0",
        "source_stage": "stage4_a160w4_3p0m_150e",
        "source_run_dir": "chroma420_a320_chroma420_a320_w4_3p0m_bunny",
        "experiment": "chroma420_a320",
        "modelsize": "3.0",
        "split_stage": "a320",
        "branch_width": 4,
    },
]


STAGE5A_MODEL_IDS = {
    "full_rgb_1p5",
    "rgb_a160_w4_1p5",
    "chroma_a160_w4_1p5",
}

STAGE5B_MODEL_IDS = {
    "full_rgb_0p75",
    "rgb_a160_w4_0p75",
    "chroma_a160_w4_0p75",
    "full_rgb_3p0",
    "rgb_a160_w8_3p0",
    "chroma_a160_w8_3p0",
    "rgb_a320_w4_3p0",
    "chroma_a320_w4_3p0",
}


FULL_UNIFORM_CONFIGS = [
    {
        "tag": "U8E6",
        "args": ["--quant_scheme", "uniform", "--quant_model_bit", "8"],
    },
    {
        "tag": "U6E6",
        "args": ["--quant_scheme", "uniform", "--quant_model_bit", "6"],
    },
    {
        "tag": "U4E6",
        "args": ["--quant_scheme", "uniform", "--quant_model_bit", "4"],
    },
]


RGB_COMPONENT_CONFIGS = [
    {
        "tag": "S8RGB6E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_rgb_bit", "6",
        ],
    },
    {
        "tag": "S6RGB4E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "6",
            "--quant_rgb_bit", "4",
        ],
    },
]


CHROMA_COMPONENT_CONFIGS = [
    {
        "tag": "S8Y8C6E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "8",
            "--quant_chroma_bit", "6",
        ],
    },
    {
        "tag": "S8Y8C4E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "8",
            "--quant_chroma_bit", "4",
        ],
    },
    {
        "tag": "S8Y8C2E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "8",
            "--quant_chroma_bit", "2",
        ],
    },
    {
        "tag": "S8Y6C4E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "4",
        ],
    },
    {
        "tag": "S8Y6C2E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "2",
        ],
    },
    {
        "tag": "S6Y6C4E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "6",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "4",
        ],
    },
    {
        "tag": "S6Y6C2E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "6",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "2",
        ],
    },
    {
        "tag": "S6Y8C2E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "6",
            "--quant_y_bit", "8",
            "--quant_chroma_bit", "2",
        ],
    },
]


CHROMA_C3_CONFIGS = [
    {
        "tag": "S8Y8C3E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "8",
            "--quant_chroma_bit", "3",
        ],
    },
    {
        "tag": "S8Y6C3E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "8",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "3",
        ],
    },
    {
        "tag": "S6Y6C3E6",
        "args": [
            "--quant_scheme", "component",
            "--quant_model_bit", "8",
            "--quant_shared_bit", "6",
            "--quant_y_bit", "6",
            "--quant_chroma_bit", "3",
        ],
    },
]


SMOKE_JOBS = {
    ("full_rgb_1p5", "U8E6"),
    ("rgb_a160_w4_1p5", "U8E6"),
    ("chroma_a160_w4_1p5", "U8E6"),
    ("chroma_a160_w4_1p5", "S8Y8C4E6"),
    ("chroma_a160_w4_1p5", "S8Y6C2E6"),
}


def normalize_key(name: str) -> str:
    name = name.replace("blocks.0.", "")
    if name.startswith("module."):
        name = name[len("module."):]
    return name


def validate_checkpoint_signature(checkpoint_path: Path, spec: Dict) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a training checkpoint with a state_dict.")

    state = {
        normalize_key(str(key)): value
        for key, value in checkpoint["state_dict"].items()
    }
    keys = set(state)

    experiment = spec["experiment"]
    if experiment == "rgb444_hnerv":
        required_prefixes = ["decoder.", "head_layer."]
        forbidden_prefixes = ["shared_decoder.", "rgb_branch.", "y_branch.", "cbcr_branch."]
        head_key = "head_layer.weight"
    elif experiment.startswith("rgbsplit_"):
        required_prefixes = ["shared_decoder.", "rgb_branch.", "rgb_head."]
        forbidden_prefixes = ["y_branch.", "cbcr_branch.", "head_layer.", "decoder."]
        head_key = "rgb_head.weight"
    elif experiment.startswith("chroma420_"):
        # A320 has an Identity cbcr_branch and therefore no cbcr_branch parameters.
        required_prefixes = ["shared_decoder.", "y_branch.", "y_head.", "cbcr_head."]
        if spec["split_stage"] == "a160":
            required_prefixes.append("cbcr_branch.")
        forbidden_prefixes = ["rgb_branch.", "rgb_head.", "head_layer.", "decoder."]
        head_key = "y_head.weight"
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing_prefixes:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not match {experiment}; "
            f"missing prefixes: {missing_prefixes}"
        )

    forbidden_hits = [
        prefix
        for prefix in forbidden_prefixes
        if any(key.startswith(prefix) for key in keys)
    ]
    if forbidden_hits:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has incompatible prefixes for {experiment}: "
            f"{forbidden_hits}"
        )

    if spec["branch_width"] is not None:
        if head_key not in state:
            raise ValueError(f"Missing {head_key} in {checkpoint_path}.")
        actual_width = int(state[head_key].shape[1])
        expected_width = int(spec["branch_width"])
        if actual_width != expected_width:
            raise ValueError(
                f"{checkpoint_path}: expected branch width {expected_width}, "
                f"but {head_key} has input width {actual_width}."
            )

    return {
        "epoch": checkpoint.get("epoch", None),
        "state_tensor_count": len(state),
        "head_key": head_key,
        "head_shape": tuple(state[head_key].shape) if head_key in state else None,
    }


def locate_checkpoint(drive_root: Path, spec: Dict) -> Tuple[Path, Dict]:
    run_root = drive_root / "output" / spec["source_stage"] / spec["source_run_dir"]
    if not run_root.is_dir():
        raise FileNotFoundError(f"Expected run folder is missing: {run_root}")

    candidates = sorted(run_root.rglob("epoch150.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one epoch150.pth below {run_root}, found {len(candidates)}:\n"
            + "\n".join(str(path) for path in candidates)
        )

    selected = candidates[0]
    info = validate_checkpoint_signature(selected, spec)
    return selected, info


def build_manifest(drive_root: Path, manifest_path: Path) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        selected, info = locate_checkpoint(drive_root, spec)
        row = {
            **spec,
            "checkpoint_path": str(selected),
            "checkpoint_filename": selected.name,
            "checkpoint_epoch": info["epoch"],
            "state_tensor_count": info["state_tensor_count"],
            "head_key": info["head_key"],
            "head_shape": str(info["head_shape"]),
        }
        rows.append(row)

        print("\n" + "-" * 100)
        print(spec["model_id"])
        print("  source stage:", spec["source_stage"])
        print("  source run:  ", spec["source_run_dir"])
        print("  checkpoint:  ", selected)
        print("  epoch:       ", info["epoch"])
        print("  head shape:  ", info["head_shape"])

    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print("\nSaved checkpoint manifest:", manifest_path)
    return manifest


def configs_for_experiment(experiment: str, c3_only: bool = False) -> List[Dict]:
    if c3_only:
        return CHROMA_C3_CONFIGS if experiment.startswith("chroma420_") else []
    if experiment == "rgb444_hnerv":
        return FULL_UNIFORM_CONFIGS
    if experiment.startswith("rgbsplit_"):
        return FULL_UNIFORM_CONFIGS + RGB_COMPONENT_CONFIGS
    if experiment.startswith("chroma420_"):
        return FULL_UNIFORM_CONFIGS + CHROMA_COMPONENT_CONFIGS
    raise ValueError(f"Unsupported experiment: {experiment}")


def load_completed(raw_csv: Path) -> set:
    if not raw_csv.is_file():
        return set()
    frame = pd.read_csv(raw_csv)
    required = {"checkpoint_path", "quant_tag"}
    if not required.issubset(frame.columns):
        return set()

    completed = set()
    for _, row in frame.iterrows():
        checkpoint = str(row["checkpoint_path"])
        tag = str(row["quant_tag"])
        if checkpoint and tag and tag.lower() != "nan":
            completed.add((checkpoint, tag))
    return completed


def stream_command(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command:")
    print(shlex.join(list(command)))

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def base_evaluation_command(
    trainer: Path,
    spec: Dict,
    checkpoint_path: str,
    run_name: str,
    stage5_out: Path,
    raw_csv: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(trainer),
        "--eval_only",
        "--not_resume",
        "--overwrite",

        "-e", "150",
        "--data_path", "data/bunny",
        "--vid", "bunny",
        "--outf", str(stage5_out),

        "--conv_type", "convnext", "pshuffel",
        "--act", "gelu",
        "--norm", "none",
        "--crop_list", "640_1280",
        "--resize_list", "-1",
        "--loss", "L2",
        "--enc_strds", "5", "4", "4", "2", "2",
        "--enc_dim", "64_16",
        "--dec_strds", "5", "4", "4", "2", "2",
        "--ks", "0_1_5",
        "--reduce", "1.2",
        "--lower_width", "12",
        "--eval_freq", "30",
        "-b", "2",
        "--lr", "0.001",
        "--manualSeed", "1",

        "--experiment", spec["experiment"],
        "--run_name", run_name,
        "--modelsize", str(spec["modelsize"]),
        "--weight", checkpoint_path,
        "--results_csv", str(raw_csv),

        "--quant_axis", "0",
        "--quant_embed_bit", "6",
        "--quant_storage_mode", "packed",
        "--save_packed_quant",
    ]

    branch_width = spec.get("branch_width")
    split_stage = spec.get("split_stage")

    # Pandas converts empty manifest cells to NaN. Treat NaN exactly like None
    # for full HNeRV models, which do not have split-stage or branch-width args.
    has_branch_width = (
        branch_width is not None
        and not pd.isna(branch_width)
    )
    has_split_stage = (
        split_stage is not None
        and not pd.isna(split_stage)
        and str(split_stage).lower() != "nan"
    )

    if has_branch_width:
        if not has_split_stage:
            raise ValueError(
                f"Model {spec.get('model_id', '<unknown>')} has branch_width={branch_width} "
                "but no valid split_stage."
            )
        command.extend([
            "--split_stage", str(split_stage),
            "--branch_width", str(int(branch_width)),
        ])

    if spec["experiment"].startswith("chroma420_"):
        command.extend([
            "--lambda_y", "1.0",
            "--lambda_c", "1.0",
            "--lambda_rgb", "0.1",
            "--chroma_scale", "2",
            "--chroma_downsample", "area",
            "--chroma_upsample", "bilinear",
        ])

    return command


def selected_for_phase(model_id: str, experiment: str, phase: str) -> bool:
    if phase == "smoke":
        return True
    if phase == "stage5a":
        return model_id in STAGE5A_MODEL_IDS
    if phase == "stage5b":
        return model_id in STAGE5B_MODEL_IDS
    if phase == "all":
        return True
    if phase == "c3":
        return model_id == "chroma_a160_w4_1p5"
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["inspect", "smoke", "stage5a", "stage5b", "all", "c3"],
        required=True,
    )
    parser.add_argument("--repo_root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--stage5_name", default=DEFAULT_STAGE5_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    drive_root = Path(args.drive_root)
    trainer = repo_root / "train_chroma_hnerv.py"
    summarizer = repo_root / "summarize_component_quant.py"

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")
    if not trainer.is_file():
        raise FileNotFoundError(f"Updated trainer not found: {trainer}")
    if not summarizer.is_file():
        raise FileNotFoundError(f"Summarizer not found: {summarizer}")
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root not found: {drive_root}")

    stage5_out = drive_root / "output" / args.stage5_name
    results_root = drive_root / "results"
    summary_root = results_root / args.stage5_name
    raw_csv = results_root / "chroma_hnerv_stage5_component_quant_raw.csv"
    manifest_path = results_root / "chroma_hnerv_stage5_checkpoint_manifest.csv"
    log_root = stage5_out / "logs"

    stage5_out.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(drive_root, manifest_path)

    if args.phase == "inspect":
        print("\nInspection complete. No evaluations were launched.")
        return

    completed = set() if args.force else load_completed(raw_csv)

    jobs: List[Tuple[pd.Series, Dict]] = []
    for _, row in manifest.iterrows():
        model_id = str(row["model_id"])
        experiment = str(row["experiment"])

        if not selected_for_phase(model_id, experiment, args.phase):
            continue

        configs = configs_for_experiment(
            experiment,
            c3_only=(args.phase == "c3"),
        )

        for config in configs:
            if args.phase == "smoke" and (model_id, config["tag"]) not in SMOKE_JOBS:
                continue
            jobs.append((row, config))

    print(f"\nScheduled jobs for phase '{args.phase}': {len(jobs)}")

    for index, (row, config) in enumerate(jobs, start=1):
        checkpoint_path = str(row["checkpoint_path"])
        tag = config["tag"]
        completion_key = (checkpoint_path, tag)

        if completion_key in completed:
            print(f"[{index}/{len(jobs)}] SKIP completed: {row['model_id']} | {tag}")
            continue

        run_name = f"stage5_ptq_{row['model_id']}_{tag}"
        command = base_evaluation_command(
            trainer=trainer,
            spec=row.to_dict(),
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            stage5_out=stage5_out,
            raw_csv=raw_csv,
        )
        command.extend(["--quant_tag", tag, *config["args"]])

        print("\n" + "=" * 110)
        print(f"[{index}/{len(jobs)}] {row['model_id']} | {tag}")
        print("Checkpoint:", checkpoint_path)
        print("=" * 110)

        log_path = log_root / f"{row['model_id']}_{tag}.log"
        stream_command(command, log_path)
        completed.add(completion_key)

    print("\nPhase complete.")
    print("Raw CSV:", raw_csv)
    print("Run output:", stage5_out)
    print("Logs:", log_root)
    print("\nSummarize with:")
    print(
        shlex.join([
            sys.executable,
            str(summarizer),
            str(raw_csv),
            "--output_dir",
            str(summary_root),
        ])
    )


if __name__ == "__main__":
    main()
