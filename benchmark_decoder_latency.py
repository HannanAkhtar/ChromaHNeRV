#!/usr/bin/env python3
"""Decoder-only FP32 HoneyBee latency benchmark for six archived checkpoints."""
import argparse
import csv
import hashlib
import json
import os
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


def assert_idle_gpu(allow_shared=False):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        if allow_shared:
            return f"GPU process check unavailable: {exc}"
        raise RuntimeError("Cannot verify an idle GPU with nvidia-smi") from exc
    other = sorted({int(line.strip()) for line in result.splitlines()
                    if line.strip().isdigit() and int(line.strip()) != os.getpid()})
    if other and not allow_shared:
        raise RuntimeError(f"Other GPU compute processes detected: {other}")
    return f"other_gpu_pids={other}"


def gpu_telemetry():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


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


def discover_manifest(roots):
    """Find the six original HoneyBee runs without copying or rebuilding weights."""
    manifest = {}
    for root in map(Path, roots):
        for checkpoint in root.rglob("epoch150.pth"):
            paths = {"checkpoint": checkpoint, "config": checkpoint.parent / "config.json",
                     "command": checkpoint.parent / "command.txt"}
            if not all(path.is_file() for path in paths.values()):
                continue
            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            if str(config.get("vid", "")).lower() != "honeybee" or float(config.get("modelsize", 0)) != 1.5:
                continue
            experiment = config.get("experiment")
            width = config.get("branch_width")
            name = experiment if experiment == "rgb444_hnerv" else f"{experiment}_w{int(width)}" if width is not None else ""
            if name not in ARCHITECTURES:
                continue
            if name in manifest:
                raise ValueError(f"Duplicate original HoneyBee checkpoint for {name}: {checkpoint}")
            manifest[name] = {**{field: str(path.resolve()) for field, path in paths.items()},
                              "sha256": {field: sha256(path) for field, path in paths.items()}}
    missing = sorted(set(ARCHITECTURES) - set(manifest))
    if missing:
        raise FileNotFoundError(f"Original HoneyBee checkpoints missing: {missing}; searched {list(map(str, roots))}")
    validate_manifest(manifest)
    return manifest


def cached_decoder(model, frame):
    with __import__("torch").inference_mode():
        encoded_input = model.pe_embed(frame[:, None]).float() if "pe" in getattr(model, "embed", "") else frame
        embedding = model.encoder(encoded_input)
        reference = model(frame)[0]
        decoded = model(frame, input_embed=embedding)[0]
        if not __import__("torch").allclose(reference, decoded, atol=1e-5, rtol=1e-5):
            raise ValueError("Cached embedding does not reproduce ordinary forward")
    return embedding


def precompute_embeddings(model, dataset):
    """Decode-equivalence check on representative frames; cache all 132 embeddings."""
    import torch
    embeddings = []
    first_frame = None
    for index in range(len(dataset)):
        frame = dataset[index]["img"].unsqueeze(0).cuda().float()
        if first_frame is None:
            first_frame = frame
        if index in (0, len(dataset) // 2, len(dataset) - 1):
            embedding = cached_decoder(model, frame)
        else:
            with torch.inference_mode():
                encoded_input = model.pe_embed(frame[:, None]).float() if "pe" in model.embed else frame
                embedding = model.encoder(encoded_input)
        embeddings.append(embedding)
    return first_frame, embeddings


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


def measure(model, frame, embeddings, warmup, repeats, trials):
    import torch
    observations = []
    model.measure_latency = False
    def decode(embedding):
        return model(frame, input_embed=embedding)[0]
    with torch.inference_mode():
        for index in range(warmup):
            decode(embeddings[index % len(embeddings)])
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for trial in range(trials):
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(repeats)]
            for repeat, (start, end) in enumerate(events):
                embedding = embeddings[(trial * repeats + repeat) % len(embeddings)]
                start.record()
                output = decode(embedding)
                end.record()
                if output.shape != (1, 3, 960, 1920):
                    raise ValueError("Invalid decoder output shape")
            torch.cuda.synchronize()
            for repeat, (start, end) in enumerate(events):
                elapsed = start.elapsed_time(end)
                if not 0 < elapsed < float("inf"):
                    raise ValueError("Invalid decoder timing")
                observations.append({"trial": trial, "repeat": repeat, "latency_ms": elapsed})
        peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return observations, peak


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Optional JSON mapping six original runs and artifact hashes")
    parser.add_argument("--archive-root", action="append", default=[],
                        help="Search extracted original archives recursively; repeat for multiple roots")
    parser.add_argument("--frames", required=True, help="132 original HoneyBee RGB frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Override idle-GPU check; results may be contaminated")
    args = parser.parse_args(argv)
    if not args.manifest and not args.archive_root:
        parser.error("Supply --manifest or --archive-root")
    manifest = (json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest
                else discover_manifest(args.archive_root))
    validate_manifest(manifest)
    import torch
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Primary latency benchmark requires an NVIDIA A100")
    if min(args.warmup, args.repeats, args.trials) < 1:
        parser.error("warmup, repeats and trials must all be positive")
    idle_check = assert_idle_gpu(args.allow_shared_gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from model_all import VideoDataSet
    from train_chroma_hnerv import build_model
    from uvg_utils import discover_image_files
    from PIL import Image
    files = discover_image_files(args.frames)
    if len(files) != 132 or any(Image.open(path).size != (1920, 1080) for path in files):
        raise ValueError("HoneyBee input must contain exactly 132 original 1080x1920 frames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase2_latency_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, summary, telemetry = [], [], []
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in ARCHITECTURES:
        assert_idle_gpu(args.allow_shared_gpu)
        record = manifest[name]
        cfg = reconstructed_args(record, args.frames)
        dataset = VideoDataSet(cfg)
        model = build_model(cfg).cuda().float().eval()
        checkpoint = torch.load(record["checkpoint"], map_location="cpu")
        if checkpoint.get("epoch") != 150:
            raise ValueError(f"{name}: expected epoch150 checkpoint")
        config = json.loads(Path(record["config"]).read_text(encoding="utf-8"))
        if int(config.get("manualSeed", -1)) != 1 or float(config.get("modelsize", 0)) != 1.5:
            raise ValueError(f"{name}: expected seed 1 and nominal 1.5M budget")
        architecture = checkpoint.get("architecture_config", {})
        if architecture and architecture.get("experiment") != config.get("experiment"):
            raise ValueError(f"{name}: checkpoint architecture disagrees with config")
        state = checkpoint["state_dict"]
        normalized = {key[7:] if key.startswith("module.") else key: value
                      for key, value in state.items()}
        if len(normalized) != len(state):
            raise ValueError(f"{name}: colliding checkpoint keys after module-prefix removal")
        model.load_state_dict(normalized, strict=True)
        frame, embeddings = precompute_embeddings(model, dataset)
        before = gpu_telemetry()
        observed, peak = measure(model, frame, embeddings, args.warmup, args.repeats, args.trials)
        telemetry.append({"architecture": name, "before": before, "after": gpu_telemetry()})
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
                        "decoder_gflops": config.get("decoder_gflops", float("nan")),
                        "checkpoint_sha256": record["sha256"]["checkpoint"],
                        "config_sha256": record["sha256"]["config"],
                        "command_sha256": record["sha256"]["command"],
                        "code_sha": code_sha})
        del model, embeddings, frame
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
        "cuda": torch.version.cuda, "dtype": "float32", "batch": 1, "frames": 132,
        "warmup": args.warmup, "repeats": args.repeats, "trials": args.trials,
        "code_sha": code_sha, "idle_check": idle_check,
        "allow_shared_gpu": args.allow_shared_gpu, "telemetry": telemetry,
        "note": "Sequential model order; CUDA event time excludes encoder, embedding extraction, data I/O and metrics. "
                "Events are preallocated per trial and synchronized once per trial. "
                "Peak allocated delta is process-local, measured after warm-up and excludes resident model and embedding; "
                "GPU clocks/temperature and other processes can affect elapsed time. Run on an otherwise idle A100."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
