from train_chroma_hnerv import build_parser
from uvg_utils import UVG_HNERV_PRESET, apply_dataset_preset


def parse(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    return apply_dataset_preset(args, parser, argv)


def test_uvg_preset_values():
    args = parse(["--dataset_preset", "uvg_hnerv"])
    for key, value in UVG_HNERV_PRESET.items():
        assert getattr(args, key) == value


def test_explicit_arguments_override_preset():
    args = parse([
        "--dataset_preset", "uvg_hnerv",
        "--crop_list", "720_1280",
        "--dec_strds", "5", "4", "4", "2", "2",
        "--batchSize", "1",
    ])
    assert args.crop_list == "720_1280"
    assert args.dec_strds == [5, 4, 4, 2, 2]
    assert args.batchSize == 1
    assert args.enc_strds == [5, 4, 4, 3, 2]


def test_no_preset_preserves_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    apply_dataset_preset(args, parser, [])
    assert args.crop_list == "640_1280"
    assert args.dec_strds == [5, 3, 2, 2, 2]
    assert args.batchSize == 1
