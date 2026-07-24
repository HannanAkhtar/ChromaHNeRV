import torch

from uvg_utils import atomic_torch_save


def test_atomic_save_replaces_destination(tmp_path):
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save({"value": torch.tensor([1])}, destination)
    atomic_torch_save({"value": torch.tensor([2])}, destination)
    assert torch.load(destination)["value"].item() == 2
    assert not (tmp_path / "checkpoint.pth.tmp").exists()
