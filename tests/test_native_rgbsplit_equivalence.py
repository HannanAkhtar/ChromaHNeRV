from copy import deepcopy

import torch

from model_all import HNeRV, decoder_channel_schedule
from model_chroma_hnerv import RGBSplitHNeRV, convert_full_state_dict_to_native_rgbsplit
from train_chroma_hnerv import build_parser, configure_model_size_args


def make_args(modelsize, mode="native", width=8):
    args = build_parser().parse_args([
        "--experiment", "rgbsplit_a160",
        "--split_stage", "a160",
        "--branch_width_mode", mode, "--branch_width", str(width),
        "--modelsize", str(modelsize),
        "--enc_strds", "5", "4", "4", "2", "2",
        "--dec_strds", "5", "4", "4", "2", "2",
        "--enc_dim", "64_16",
        "--ks", "0_1_5",
        "--reduce", "1.2",
        "--lower_width", "12",
        "--conv_type", "convnext", "pshuffel",
    ])
    args.final_size = 640 * 1280
    args.full_data_length = 132
    configure_model_size_args(args)
    return args


def decoder_parameter_count(model):
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if not name.startswith("encoder."))


@torch.no_grad()
def test_native_rgbsplit_equivalence():
    torch.manual_seed(17)
    for modelsize in (0.35, 0.75, 1.5, 3.0):
        args = make_args(modelsize)
        full = HNeRV(deepcopy(args)).eval()
        split = RGBSplitHNeRV(deepcopy(args)).eval()
        converted = convert_full_state_dict_to_native_rgbsplit(
            full.state_dict(), args.dec_strds, args.num_blks, args.split_stage)
        split.load_state_dict(converted, strict=True)

        assert decoder_parameter_count(full) == decoder_parameter_count(split)
        expected_schedule = tuple(decoder_channel_schedule(args))
        assert split.native_channel_schedule == expected_schedule
        assert split.shared_channel_schedule + split.rgb_branch_channels == expected_schedule

        embed_channels = int(args.enc_dim.split("_")[1])
        embedding = torch.randn(1, embed_channels, 2, 4)
        full_output = full(torch.zeros(1), input_embed=embedding)[0]
        split_output = split(torch.zeros(1), input_embed=embedding)[0]
        assert full_output.shape == split_output.shape == (1, 3, 640, 1280)
        assert (full_output - split_output).abs().max().item() < 1e-6


def test_fixed_width_shape_regression():
    for width in (4, 8):
        args = make_args(1.5, mode="fixed", width=width)
        explicit = RGBSplitHNeRV(deepcopy(args))
        delattr(args, "branch_width_mode")
        legacy_default = RGBSplitHNeRV(deepcopy(args))
        explicit_shapes = {name: tuple(value.shape) for name, value in explicit.state_dict().items()}
        legacy_shapes = {name: tuple(value.shape) for name, value in legacy_default.state_dict().items()}
        assert explicit_shapes == legacy_shapes


if __name__ == "__main__":
    test_native_rgbsplit_equivalence()
    test_fixed_width_shape_regression()
    print("native RGBSplit equivalence tests passed")
