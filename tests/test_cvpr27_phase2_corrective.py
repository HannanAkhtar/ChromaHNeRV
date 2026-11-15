import csv
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest


def test_decoder_gflops_excludes_encoder():
    torch = pytest.importorskip("torch")
    from train_chroma_hnerv import estimate_gflops

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = ""
            self.encoder = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.decoder = torch.nn.Conv2d(4, 3, 1)

        def forward(self, frame, input_embed=None):
            encoded = self.encoder(frame) if input_embed is None else input_embed
            return self.decoder(encoded), [encoded], 0

    class Args:
        embed = ""
        transform_func = staticmethod(lambda image: (image, image, torch.ones_like(image)))

    sample = [{"img": torch.rand(1, 3, 8, 8), "norm_idx": torch.zeros(1)}]
    model = Tiny()
    full = estimate_gflops(model, sample, Args(), torch.device("cpu"))
    decoder = estimate_gflops(model, sample, Args(), torch.device("cpu"), decoder_only=True)
    assert decoder == pytest.approx(3 * 8 * 8 * 4 * 2 / 1e9)
    assert full - decoder == pytest.approx(4 * 8 * 8 * 3 * 3 * 3 * 2 / 1e9)


def test_quantization_group_coverage_and_matched_rgb_architecture():
    torch = pytest.importorskip("torch")
    from train_chroma_hnerv import (build_parser, classify_quant_group, quant_model,
                                    effective_quant_bits)
    from model_chroma_hnerv import RGBSplitHNeRV, YCbCrSplit444HNeRV
    args = build_parser().parse_args(["--embed", "pe_1.25_4", "--dec_strds", "2", "2",
                                     "--fc_hw", "2_2", "--crop_list", "8_8",
                                     "--branch_width", "8", "--modelsize", "0.02"])
    args.fc_dim = 19
    baseline = RGBSplitHNeRV(args)
    args.experiment = "rgbsplit_yc_a320"
    matched = RGBSplitHNeRV(args)
    assert {key: tuple(value.shape) for key, value in baseline.state_dict().items()} == {
        key: tuple(value.shape) for key, value in matched.state_dict().items()}
    for experiment, model in (("rgbsplit_yc_a320", matched),
                              ("ycbcrsplit444_a320", YCbCrSplit444HNeRV(args))):
        args.experiment = experiment
        _, records, report = quant_model(model, args)
        decoder = {name for name, _ in model.named_parameters() if not name.startswith("encoder.")}
        assert set(records) == decoder == set(report["parameter_groups"])
        assert all(record["bits"] == 8 for record in records.values())
        assert effective_quant_bits(args)["shared"] == 8
        if experiment == "ycbcrsplit444_a320":
            assert {classify_quant_group(model, name, args) for name in decoder} == {
                "shared", "y", "chroma"}


def test_ycbcr444_individual_loss_terms_and_gradients():
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F
    from train_chroma_hnerv import (area_downsample_chroma, build_parser,
                                    compute_train_loss, rgb_to_ycbcr_bt709)
    from hnerv_utils import ycbcr_to_rgb_bt709
    args = build_parser().parse_args(["--experiment", "ycbcrsplit444_a320"])
    args.vid = "synthetic"
    ycbcr = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    rgb = ycbcr_to_rgb_bt709(ycbcr)
    loss, parts = compute_train_loss(rgb, target, torch.ones_like(rgb), args,
                                     {"rgb": rgb, "ycbcr": ycbcr})
    target_ycbcr = rgb_to_ycbcr_bt709(target)
    expected_y = F.mse_loss(ycbcr[:, :1], target_ycbcr[:, :1])
    expected_c = F.mse_loss(area_downsample_chroma(ycbcr[:, 1:3], 2),
                            area_downsample_chroma(target_ycbcr[:, 1:3], 2))
    expected_rgb = F.mse_loss(rgb, target)
    assert torch.allclose(parts["loss_y"], expected_y)
    assert torch.allclose(parts["loss_c"], expected_c)
    assert torch.allclose(parts["loss_rgb"], expected_rgb)
    assert torch.allclose(loss, expected_y + expected_c + 0.1 * expected_rgb)
    loss.backward()
    assert ycbcr.grad is not None and torch.isfinite(ycbcr.grad).all()
    assert ycbcr.grad[:, 1:3].abs().sum() > 0


def test_archive_restart_and_path_safety(tmp_path):
    from phase2_archive import extract_verified, sha256
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("Sequence/frame.png", b"original")
    destination = tmp_path / "out"
    _, audit = extract_verified(archive, destination, sha256(archive))
    assert audit["restored_files"] == 1
    (destination / "Sequence/frame.png").write_bytes(b"corrupt")
    _, audit = extract_verified(archive, destination, sha256(archive))
    assert audit["restored_files"] == 1
    assert (destination / "Sequence/frame.png").read_bytes() == b"original"
    _, audit = extract_verified(archive, destination, sha256(archive))
    assert audit["restored_files"] == 0
    with pytest.raises(ValueError, match="provenance"):
        extract_verified(archive, destination, "0" * 64)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as file:
        file.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        extract_verified(unsafe, destination)


def test_preflight_checks_transformed_geometry(tmp_path):
    from PIL import Image
    from run_cvpr27_phase2 import preflight
    for index in range(132):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / f"{index:03d}.png")
    command = ["python", "train_chroma_hnerv.py", "--data_path", str(tmp_path),
               "--crop_list", "8_8", "--resize_list", "-1", "--expected_frames", "132"]
    metadata = {"frames": 132, "crop_height": 8, "crop_width": 8,
                "source_height": 8, "source_width": 8}
    preflight({"sequence": "Synthetic"}, tmp_path, metadata, command)
    command[command.index("--resize_list") + 1] = "4_4"
    with pytest.raises(ValueError, match="transformed frames"):
        preflight({"sequence": "Synthetic"}, tmp_path, metadata, command)


@pytest.mark.parametrize("variant", ["rgbsplit_yc_a320", "ycbcrsplit444_a320"])
def test_full_one_epoch_quantized_evaluation_and_resume(tmp_path, monkeypatch, variant):
    torch = pytest.importorskip("torch")
    from PIL import Image
    import train_chroma_hnerv as training

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "build_lpips_model", lambda device: None)
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(2):
        image = np.full((8, 8, 3), 64 + index * 96, dtype=np.uint8)
        Image.fromarray(image).save(frames / f"{index:03d}.png")
    run = tmp_path / variant
    backup = tmp_path / "backup"

    def args_for(epochs):
        args = training.build_parser().parse_args([
            "--data_path", str(frames), "--vid", "Synthetic", "--experiment", variant,
            "--embed", "pe_1.25_4", "--dec_strds", "2", "2", "--fc_hw", "2_2",
            "--crop_list", "8_8", "--resize_list", "-1", "--modelsize", "0.02",
            "--lower_width", "8", "--branch_width", "8", "--workers", "0",
            "--batchSize", "1", "--epochs", str(epochs), "--eval_freq", "1",
            "--expected_frames", "2", "--run_dir", str(run), "--run_name", variant,
            "--backup_root", str(backup), "--backup_every_epochs", "1",
            "--strict_backup", "--print-freq", "100"])
        training.setup_output_dir(args)
        args.ngpus_per_node = 0
        args.lambda_y = args.lambda_c = 1.0
        args.lambda_rgb = 0.1
        return args

    training.train(None, args_for(1))
    csv_path = run / "epoch1.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["experiment"] == variant
    assert row["quant_rgb_psnr"] not in ("", "nan")
    assert float(row["decoder_gflops"]) > 0
    assert float(row["decoder_playback_params"]) > 0
    assert (run / "quant_vid.pth").is_file()
    assert (run / "completion.json").is_file()
    assert json.loads((run / "completion.json").read_text())["epoch"] == 1
    from run_cvpr27_phase2 import validate_phase2_completion
    validate_phase2_completion(run, {"run_id": variant, "variant": variant}, 1)
    checkpoint = torch.load(run / "epoch1.pth", map_location="cpu")
    assert checkpoint["epoch"] == 1 and checkpoint["optimizer"]["state"]
    assert (backup / variant / "model_latest.pth").is_file()
    training.train(None, args_for(2))
    resumed = torch.load(run / "epoch2.pth", map_location="cpu")
    assert resumed["epoch"] == 2 and resumed["optimizer"]["state"]
    assert json.loads((run / "completion.json").read_text())["epoch"] == 2
    validate_phase2_completion(run, {"run_id": variant, "variant": variant}, 2)


def test_backup_restore_keeps_newer_local_epoch(tmp_path):
    torch = pytest.importorskip("torch")
    from uvg_utils import restore_run_from_backup
    local, backup = tmp_path / "local", tmp_path / "backup"
    local.mkdir()
    backup.mkdir()
    torch.save({"epoch": 2}, local / "model_latest.pth")
    torch.save({"epoch": 1}, backup / "model_latest.pth")
    os.utime(local / "model_latest.pth", (10, 10))
    os.utime(backup / "model_latest.pth", (20, 20))
    result = restore_run_from_backup(local, backup, 2)
    assert "model_latest.pth" in result["kept_local"]
    assert torch.load(local / "model_latest.pth", map_location="cpu")["epoch"] == 2


def test_discover_original_latency_manifest(tmp_path):
    from benchmark_decoder_latency import ARCHITECTURES, discover_manifest
    for name in ARCHITECTURES:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "epoch150.pth").write_bytes(name.encode())
        experiment, _, width = name.partition("_w")
        (directory / "config.json").write_text(json.dumps({
            "vid": "HoneyBee", "modelsize": 1.5, "experiment": experiment,
            "branch_width": int(width) if width else None}))
        (directory / "command.txt").write_text("original command")
    manifest = discover_manifest([tmp_path])
    assert set(manifest) == set(ARCHITECTURES)
    assert all(len(row["sha256"]["checkpoint"]) == 64 for row in manifest.values())


def test_merger_retains_bosphorus_mixed_seed_without_quant_fallback(tmp_path):
    pandas = pytest.importorskip("pandas")
    from merge_cvpr27_phase2 import main
    references = tmp_path / "references.csv"
    common = {"vid": "Bosphorus", "modelsize": 1.5, "cur_epoch": 150,
              "quant_model_bit": 8, "quant_embed_bit": 6, "loss": "Fusion6",
              "rgb_psnr": 30.0, "quant_rgb_psnr": 29.0,
              "decoder_params_M": 1.4, "decoder_gflops": 2.0}
    pandas.DataFrame([{**common, "experiment": "rgbsplit_a320", "manualSeed": 3},
                      {**common, "experiment": "chroma420_a320", "manualSeed": 1,
                       "rgb_psnr": 31.0, "quant_rgb_psnr": float("nan")}]).to_csv(references, index=False)
    new_root = tmp_path / "new"
    for variant in ("rgbsplit_yc_a320", "ycbcrsplit444_a320"):
        run = new_root / f"cvpr27_phase2_Bosphorus_{variant}_1p5m_a320_w8_seed1"
        run.mkdir(parents=True)
        (run / "completion.json").write_text(json.dumps({"status": "complete", "epoch": 150}))
        pandas.DataFrame([{**common, "experiment": variant, "manualSeed": 1,
                           "rgb_psnr": 32.0}]).to_csv(run / "epoch150.csv", index=False)
    output = tmp_path / "merged"
    main(["--new-root", str(new_root), "--references", str(references),
          "--output", str(output)])
    pairs = pandas.read_csv(output / "phase2_ablation_pairs.csv")
    mixed = pairs[(pairs.vid == "Bosphorus") &
                  (pairs.comparison == "rgbsplit_yc_a320") & (pairs.precision == "fp32")]
    assert len(mixed) == 1 and bool(mixed.iloc[0].mixed_seed)
    assert mixed.iloc[0].delta_rgb_psnr == pytest.approx(2.0)
    strict = pandas.read_csv(output / "phase2_ablation_seed_matched_summary.csv")
    assert strict.empty
    per_sequence = pandas.read_csv(output / "phase2_ablation_per_sequence.csv")
    assert per_sequence[(per_sequence.experiment == "chroma420_a320") &
                        (per_sequence.precision == "m8e6")].empty
