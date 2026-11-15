import json
from pathlib import Path

import pytest

from benchmark_decoder_latency import ARCHITECTURES, validate_manifest
from run_cvpr27_phase2 import SEQUENCES, VARIANTS, jobs, main


def test_exact_phase2_plan_and_strict_manifest(capsys):
    planned = main(["--dry-run"])
    assert len(planned) == 16
    assert len({job["run_id"] for job in planned}) == 16
    assert {job["sequence"] for job in planned} == set(SEQUENCES)
    assert {job["variant"] for job in planned} == set(VARIANTS)
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/cvpr27_phase2_132.json").read_text())
    assert all(row["frames"] == 132 for row in manifest.values())
    assert "Planned Phase 2 jobs: 16" in capsys.readouterr().out


def test_bunny_preflight_requires_original_data():
    with pytest.raises(FileNotFoundError, match="Bunny"):
        main(["--only-sequence", "Bunny", "--max-runs", "1"])


def test_latency_manifest_rejects_missing_artifacts():
    with pytest.raises(ValueError, match="six"):
        validate_manifest({})
    with pytest.raises(ValueError, match="missing"):
        validate_manifest({name: {} for name in ARCHITECTURES})


def test_loss_and_architecture_when_torch_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from train_chroma_hnerv import (area_downsample_chroma, build_parser,
                                    compute_train_loss, rgb_to_ycbcr_bt709)
    from model_chroma_hnerv import RGBSplitHNeRV, YCbCrSplit444HNeRV
    from torch.nn import functional as F
    args = build_parser().parse_args([])
    args.vid = "synthetic"
    args.chroma_scale = 2
    args.lambda_y = args.lambda_c = 1.0
    args.lambda_rgb = .1
    predicted = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    args.experiment = "rgbsplit_yc_a320"
    loss, terms = compute_train_loss(predicted, target, torch.ones_like(target), args)
    p, t = rgb_to_ycbcr_bt709(predicted), rgb_to_ycbcr_bt709(target)
    expected = (F.mse_loss(p[:, :1], t[:, :1]) +
                F.mse_loss(area_downsample_chroma(p[:, 1:3], 2),
                           area_downsample_chroma(t[:, 1:3], 2)) +
                .1 * F.mse_loss(predicted, target))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert predicted.grad.isfinite().all() and predicted.grad.abs().sum() > 0
    assert "loss_total" in terms
    args.experiment = "rgbsplit_a320"
    old_loss, old_terms = compute_train_loss(predicted.detach(), target,
                                              torch.ones_like(target), args)
    assert "loss_y" not in old_terms
    assert torch.isfinite(old_loss)
    assert RGBSplitHNeRV is not YCbCrSplit444HNeRV


def test_split444_full_resolution_and_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from train_chroma_hnerv import build_parser, compute_train_loss
    from model_chroma_hnerv import YCbCrSplit444HNeRV
    args = build_parser().parse_args([])
    args.embed = "pe_1.25_4"
    args.enc_strds = []
    args.dec_strds = [2, 2]
    args.fc_hw = "2_2"
    args.fc_dim = 16
    args.ks = "0_1_3"
    args.branch_width = 8
    args.split_stage = "a320"
    args.vid = "synthetic"
    args.experiment = "ycbcrsplit444_a320"
    model = YCbCrSplit444HNeRV(args)
    rgb, _, _, aux = model(torch.tensor([0.5]), return_aux=True)
    assert rgb.shape == (1, 3, 8, 8)
    assert aux["y"].shape == (1, 1, 8, 8)
    assert aux["cbcr"].shape == (1, 2, 8, 8)
    target = torch.rand_like(rgb)
    args.lambda_y = args.lambda_c = 1
    args.lambda_rgb = .1
    loss, _ = compute_train_loss(rgb, target, torch.ones_like(rgb), args, aux)
    loss.backward()
    assert any(param.grad is not None and param.grad.isfinite().all()
               for param in model.parameters())
