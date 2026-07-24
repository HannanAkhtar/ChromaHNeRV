from pathlib import Path

from uvg_utils import UVG_SEQUENCES, load_uvg_manifest


def test_uvg_manifest():
    manifest = load_uvg_manifest(Path(__file__).parents[1] / "configs" / "uvg7.json")
    assert tuple(manifest) == UVG_SEQUENCES
    assert manifest["ShakeNDry"]["frames"] == 300
    assert all(
        metadata["frames"] == 600
        for sequence, metadata in manifest.items()
        if sequence != "ShakeNDry")
