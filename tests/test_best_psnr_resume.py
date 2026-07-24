from pathlib import Path

from uvg_utils import checkpoint_best_rgb_psnr, update_best_rgb_psnr


def test_new_and_legacy_checkpoints_default_to_negative_infinity():
    assert checkpoint_best_rgb_psnr(None) == -float("inf")
    assert checkpoint_best_rgb_psnr({"epoch": 10}) == -float("inf")


def test_checkpoint_best_psnr_survives_resume():
    checkpoint = {"epoch": 30, "best_rgb_psnr": 34.25}
    assert checkpoint_best_rgb_psnr(checkpoint) == 34.25


def test_worse_resume_evaluation_is_not_best():
    is_best, best = update_best_rgb_psnr(33.0, 34.25)
    assert not is_best
    assert best == 34.25


def test_better_resume_evaluation_is_best():
    is_best, best = update_best_rgb_psnr(35.0, 34.25)
    assert is_best
    assert best == 35.0


def test_training_checkpoint_records_best_psnr():
    source = (Path(__file__).resolve().parents[1] / "train_chroma_hnerv.py").read_text()
    assert '"best_rgb_psnr": float(best_rgb_psnr)' in source
    assert "if evaluated_this_epoch and is_best_epoch:" in source
