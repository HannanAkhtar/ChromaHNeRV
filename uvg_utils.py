import json
import math
import os
import re
import shutil
from pathlib import Path


UVG_SEQUENCES = (
    "Beauty", "Bosphorus", "HoneyBee", "Jockey",
    "ReadySetGo", "ShakeNDry", "YachtRide",
)
UVG_SIZES = (0.35, 0.75, 1.5, 3.0)
UVG_WIDTHS = (8, 4)
UVG_FAMILIES = (
    "rgb444", "ycbcr444", "rgbsplit_a320",
    "chroma420_a320", "rgbsplit_a160", "chroma420_a160",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

UVG_HNERV_PRESET = {
    "crop_list": "960_1920",
    "resize_list": "-1",
    "enc_strds": [5, 4, 4, 3, 2],
    "dec_strds": [5, 4, 4, 3, 2],
    "enc_dim": "64_16",
    "ks": "0_1_5",
    "reduce": 1.2,
    "lower_width": 12,
    "conv_type": ["convnext", "pshuffel"],
    "act": "gelu",
    "norm": "none",
    "loss": "L2",
    "batchSize": 2,
    "epochs": 150,
    "eval_freq": 30,
    "lr": 0.001,
    "manualSeed": 1,
    "quant_model_bit": 8,
    "quant_embed_bit": 6,
}


def load_uvg_manifest(path):
    with open(path, encoding="utf-8") as file:
        manifest = json.load(file)
    if set(manifest) != set(UVG_SEQUENCES):
        raise ValueError(
            f"UVG manifest must contain exactly {list(UVG_SEQUENCES)}, got {sorted(manifest)}.")
    return manifest


def explicit_argument_destinations(parser, argv):
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def apply_dataset_preset(args, parser, argv):
    if args.dataset_preset == "none":
        return args
    if args.dataset_preset != "uvg_hnerv":
        raise ValueError(f"Unsupported dataset preset: {args.dataset_preset}")
    explicit = explicit_argument_destinations(parser, argv)
    for destination, value in UVG_HNERV_PRESET.items():
        if destination not in explicit:
            setattr(args, destination, list(value) if isinstance(value, list) else value)
    return args


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def discover_image_files(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {root}")
    files = [
        path for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=natural_sort_key)
    if not files:
        raise ValueError(
            f"No supported images (.png, .jpg, .jpeg) were found in sequence directory: {root}")
    return [str(path) for path in files]


def decoder_stage_resolutions(crop_height, crop_width, dec_strds):
    total_stride = math.prod(dec_strds)
    if crop_height % total_stride or crop_width % total_stride:
        raise ValueError(
            f"Output {crop_height}x{crop_width} is not divisible by decoder stride product "
            f"{total_stride}.")
    height, width = crop_height // total_stride, crop_width // total_stride
    resolutions = []
    for stride in dec_strds:
        height *= stride
        width *= stride
        resolutions.append((height, width))
    return resolutions


def split_resolution_metadata(crop_list, dec_strds, split_alias=None):
    crop_height, crop_width = [int(value) for value in crop_list.split("_")[:2]]
    stages = decoder_stage_resolutions(crop_height, crop_width, dec_strds)
    metadata = {
        "decoder_stage_resolutions": ",".join(f"{height}x{width}" for height, width in stages),
        "split_alias": split_alias or "none",
        "split_stage_index": -1,
        "shared_output_height": crop_height,
        "shared_output_width": crop_width,
        "chroma_output_height": crop_height,
        "chroma_output_width": crop_width,
        "final_output_height": crop_height,
        "final_output_width": crop_width,
    }
    if split_alias in ("a320", "a160"):
        shared_index = len(stages) - (2 if split_alias == "a320" else 3)
        shared_height, shared_width = stages[shared_index]
        chroma_index = shared_index if split_alias == "a320" else shared_index + 1
        chroma_height, chroma_width = stages[chroma_index]
        metadata.update({
            "split_stage_index": shared_index,
            "shared_output_height": shared_height,
            "shared_output_width": shared_width,
            "chroma_output_height": chroma_height,
            "chroma_output_width": chroma_width,
        })
    return metadata


def atomic_torch_save(obj, destination):
    import torch

    destination = os.fspath(destination)
    temporary = destination + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    try:
        with open(temporary, "wb") as file:
            torch.save(obj, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def size_tag(size):
    return str(float(size)).replace(".", "p")


def family_to_experiment(family):
    return {
        "rgb444": ("rgb444_hnerv", None),
        "ycbcr444": ("ycbcr444_hnerv", None),
        "rgbsplit_a320": ("rgbsplit_a320", "a320"),
        "chroma420_a320": ("chroma420_a320", "a320"),
        "rgbsplit_a160": ("rgbsplit_a160", "a160"),
        "chroma420_a160": ("chroma420_a160", "a160"),
    }[family]


def build_uvg_jobs(
        sequences=UVG_SEQUENCES, sizes=UVG_SIZES, widths=UVG_WIDTHS,
        families=UVG_FAMILIES, seed=1):
    jobs = []
    for sequence in sequences:
        for size in sizes:
            for family in families:
                experiment, split_stage = family_to_experiment(family)
                family_widths = widths if split_stage else (None,)
                for width in family_widths:
                    if split_stage:
                        architecture_label = f"{family}_w{width}"
                    else:
                        architecture_label = f"{family}_full_native"
                    run_id = (
                        f"uvg7_{sequence}_{architecture_label}_"
                        f"{size_tag(size)}m_seed{seed}")
                    jobs.append({
                        "sequence": sequence,
                        "modelsize": float(size),
                        "family": family,
                        "experiment": experiment,
                        "split_stage": split_stage,
                        "branch_width": width,
                        "branch_width_mode": "fixed" if split_stage else "none",
                        "manualSeed": int(seed),
                        "quant_model_bit": 8,
                        "quant_embed_bit": 6,
                        "run_id": run_id,
                    })
    return jobs


def completion_status(run_dir, epochs):
    run_dir = Path(run_dir)
    completion_path = run_dir / "completion.json"
    checkpoint = run_dir / f"epoch{epochs}.pth"
    final_csv = run_dir / f"epoch{epochs}.csv"
    complete = False
    if completion_path.is_file():
        try:
            with completion_path.open(encoding="utf-8") as file:
                complete = json.load(file).get("status") == "complete"
        except (json.JSONDecodeError, OSError):
            complete = False
    if complete and checkpoint.is_file() and (run_dir / "model_best.pth").is_file() and final_csv.is_file():
        return "complete"
    if (run_dir / "model_latest.pth").is_file():
        return "resume"
    return "new"


def checkpoint_best_rgb_psnr(checkpoint):
    if checkpoint is None or "best_rgb_psnr" not in checkpoint:
        return -float("inf")
    return float(checkpoint["best_rgb_psnr"])


def update_best_rgb_psnr(current_rgb_psnr, historical_best):
    is_best = current_rgb_psnr >= historical_best
    return is_best, max(historical_best, current_rgb_psnr)


def backup_artifact_names(
        epochs, final=False, launcher_log="", latest_csv=""):
    names = [
        "model_latest.pth", "model_best.pth", "rank0.txt", "config.json",
        "command.txt", "environment.txt", "git_commit.txt",
    ]
    if latest_csv:
        names.append(latest_csv)
    if launcher_log:
        names.append(launcher_log)
    if final:
        names.extend([
            f"epoch{epochs}.pth", f"epoch{epochs}.csv", "completion.json",
            "quant_vid.pth", "img_decoder.pth",
        ])
    return names


def artifact_is_valid(path):
    path = Path(path)
    if not path.is_file():
        return False
    if path.suffix == ".pth":
        try:
            import torch
            torch.load(path, map_location="cpu")
        except Exception:
            return False
    elif path.name in {"completion.json", "config.json", "failure.json"}:
        try:
            with path.open(encoding="utf-8") as file:
                json.load(file)
        except (OSError, json.JSONDecodeError):
            return False
    return True


def restore_run_from_backup(local_run_dir, backup_run_dir, epochs):
    local_run_dir = Path(local_run_dir)
    backup_run_dir = Path(backup_run_dir)
    summary = {
        "restored": [],
        "kept_local": [],
        "invalid_local": [],
        "invalid_backup": [],
    }
    if not backup_run_dir.is_dir():
        return summary

    for backup_path in backup_run_dir.rglob("*"):
        if not backup_path.is_file() or backup_path.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        relative_path = backup_path.relative_to(backup_run_dir)
        local_path = local_run_dir / relative_path
        backup_valid = artifact_is_valid(backup_path)
        local_valid = artifact_is_valid(local_path) if local_path.exists() else False

        if local_path.exists() and not local_valid:
            summary["invalid_local"].append(str(relative_path))
        if not backup_valid:
            summary["invalid_backup"].append(str(relative_path))
            if local_path.exists():
                summary["kept_local"].append(str(relative_path))
            continue

        should_restore = not local_path.exists() or not local_valid
        if local_path.exists() and local_valid:
            should_restore = (
                backup_path.stat().st_mtime > local_path.stat().st_mtime)
        if should_restore:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, local_path)
            summary["restored"].append(str(relative_path))
        else:
            summary["kept_local"].append(str(relative_path))
    return summary
