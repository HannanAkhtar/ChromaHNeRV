from pathlib import Path

from uvg_utils import UVG_SEQUENCES, load_uvg_manifest


def test_uvg_manifest():
    manifest = load_uvg_manifest(Path(__file__).parents[1] / "configs" / "uvg7.json")
    assert tuple(manifest) == UVG_SEQUENCES
    assert all(metadata["frames"] == 132 for metadata in manifest.values())
    assert all(metadata["source_height"] == 1080 for metadata in manifest.values())
    assert all(metadata["source_width"] == 1920 for metadata in manifest.values())
