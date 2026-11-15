#!/usr/bin/env python3
"""Decoder-only FP32 HoneyBee latency benchmark for six archived checkpoints."""
import argparse
import csv
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ARCHITECTURES = ("rgb444_hnerv", "rgbsplit_a320_w8", "chroma420_a320_w8",
                 "chroma420_a320_w4", "chroma420_a160_w8", "chroma420_a160_w4")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    if set(manifest) != set(ARCHITECTURES):
        raise ValueError(f"Expected exactly six HoneyBee architectures: {ARCHITECTURES}")
    for name, record in manifest.items():
        for field in ("checkpoint", "config", "command", "sha256"):
            if not record.get(field):
                raise ValueError(f"{name}: missing {field}")
        for field in ("checkpoint", "config", "command"):
            path = Path(record[field])
            if not path.is_file():
                raise FileNotFoundError(f"{name}: missing {field}: {path}")
            if field in record["sha256"] and sha256(path) != record["sha256"][field]:
                raise ValueError(f"{name}: SHA-256 mismatch for {field}")
        if set(record["sha256"]) != {"checkpoint", "config", "command"}:
            raise ValueError(f"{name}: SHA-256 required for all three artifacts")


def cached_decoder(model, frame):
    with __import__("torch").inference_mode():
        embedding = model.encoder(frame)
        reference = model(frame)[0]
        decoded = model(frame, input_embed=embedding)[0]
        if not __import__("torch").allclose(reference, decoded, atol=1e-5, rtol=1e-5):
            raise ValueError("Cached embedding does not reproduce ordinary forward")
    return embedding


def reconstructed_args(record, frames_dir):
    from train_chroma_hnerv import build_parser, configure_model_size_args
    from uvg_utils import apply_dataset_preset
    tokens = shlex.split(Path(record["command"]).read_text(encoding="utf-8"))
    tokens = tokens[tokens.index("--data_path"):] if "--data_path" in tokens else tokens[2:]
    parser = build_parser()
    args = parser.parse_args(tokens)
    apply_dataset_preset(args, parser, tokens)
    # The command records the original enc_dim; config.json can contain the calibrated value.
    args.data_path = str(frames_dir)
    args.expected_frames = 132
    args.expected_source_height = 1080
    args.expected_source_width = 1920
    args.final_size = 960 * 1920
    args.full_data_length = 132
    configure_model_size_args(args)
    config = json.loads(Path(record["config"]).read_text(encoding="utf-8"))
    if config.get("experiment") != args.experiment:
        raise ValueError("Checkpoint command and config disagree on experiment")
    return args


def measure(model, frame, embedding, warmup, repeats, trials):
    import torch
    observations = []
    model.measure_latency = False
    def decode():
        return model(frame, input_embed=embedding)[0]
    with torch.inference_mode():
        for _ in range(warmup):
            decode()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for trial in range(trials):
            for repeat in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = decode()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                if output.shape != (1, 3, 960, 1920) or not 0 < elapsed < float("inf"):
                    raise ValueError("Invalid decoder output or timing")
                observations.append({"trial": trial, "repeat": repeat, "latency_ms": elapsed})
        peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return observations, peak


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON mapping six architecture names to checkpoint/config/command/SHA-256")
    parser.add_argument("--frames", required=True, help="132 original HoneyBee RGB frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    import torch
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Primary latency benchmark requires an NVIDIA A100")
    from model_all import VideoDataSet
    from train_chroma_hnerv import build_model
    from uvg_utils import discover_image_files
    from PIL import Image
    files = discover_image_files(args.frames)
    if len(files) != 132 or any(Image.open(path).size != (1920, 1080) for path in files):
        raise ValueError("HoneyBee input must contain exactly 132 original 1080x1920 frames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in ARCHITECTURES:
        record = manifest[name]
        cfg = reconstructed_args(record, args.frames)
        dataset = VideoDataSet(cfg)
        frame = dataset[0]["img"].unsqueeze(0).cuda().float()
        model = build_model(cfg).cuda().float().eval()
        checkpoint = torch.load(record["checkpoint"], map_location="cpu")
        if checkpoint.get("epoch") != 150:
            raise ValueError(f"{name}: expected epoch150 checkpoint")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        embedding = cached_decoder(model, frame)
        observed, peak = measure(model, frame, embedding, args.warmup, args.repeats, args.trials)
        values = [item["latency_ms"] for item in observed]
        if len(values) != args.repeats * args.trials:
            raise AssertionError("Timing row count mismatch")
        rows.extend({"architecture": name, **item} for item in observed)
        ordered = sorted(values)
        summary.append({"architecture": name, "mean_ms": statistics.mean(values),
                        "median_ms": statistics.median(values), "std_ms": statistics.pstdev(values),
                        "p5_ms": ordered[int(.05 * (len(ordered)-1))],
                        "p95_ms": ordered[int(.95 * (len(ordered)-1))],
                        "fps": 1000 / statistics.mean(values),
                        "peak_allocated_delta_bytes": peak,
                        "estimated_gflops": json.loads(Path(record["config"]).read_text(encoding="utf-8")).get("estimated_gflops", float("nan")),
                        "checkpoint_sha256": record["sha256"]["checkpoint"],
                        "code_sha": code_sha})
        del model, embedding, frame
        torch.cuda.empty_cache()
    baseline = summary[0]["mean_ms"]
    for item in summary:
        item["speedup_vs_full_rgb"] = baseline / item["mean_ms"]
    for filename, data in (("phase2_latency_per_trial.csv", rows),
                           ("phase2_latency_summary.csv", summary)):
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, data[0].keys()); writer.writeheader(); writer.writerows(data)
    (output / "phase2_latency_metadata.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "dtype": "float32", "batch": 1,
        "warmup": args.warmup, "repeats": args.repeats, "trials": args.trials,
        "code_sha": code_sha, "note": "One HoneyBee sequence; allocator peak is process-local"
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
