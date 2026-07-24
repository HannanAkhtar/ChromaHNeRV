from argparse import Namespace

import pytest
from PIL import Image

from model_all import VideoDataSet
from uvg_utils import discover_image_files


def write_image(path, size=(12, 8)):
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def dataset_args(path, expected_frames=-1):
    return Namespace(
        data_path=str(path), crop_list="-1", resize_list="-1",
        expected_frames=expected_frames,
        expected_source_height=8, expected_source_width=12,
    )


def test_directory_filtering_and_numeric_order(tmp_path):
    write_image(tmp_path / "frame10.png")
    write_image(tmp_path / "frame2.jpg")
    write_image(tmp_path / "frame1.jpeg")
    (tmp_path / "desktop.ini").write_text("ignored")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / ".hidden.png").write_text("ignored")
    (tmp_path / "nested").mkdir()
    write_image(tmp_path / "nested" / "frame0.png")
    files = discover_image_files(tmp_path)
    assert [path.split("\\")[-1].split("/")[-1] for path in files] == [
        "frame1.jpeg", "frame2.jpg", "frame10.png"]
    dataset = VideoDataSet(dataset_args(tmp_path, expected_frames=3))
    assert len(dataset) == 3


def test_empty_directory_error(tmp_path):
    with pytest.raises(ValueError, match="No supported images"):
        discover_image_files(tmp_path)


def test_expected_frame_mismatch(tmp_path):
    write_image(tmp_path / "1.png")
    with pytest.raises(ValueError, match="2 were expected"):
        VideoDataSet(dataset_args(tmp_path, expected_frames=2))
