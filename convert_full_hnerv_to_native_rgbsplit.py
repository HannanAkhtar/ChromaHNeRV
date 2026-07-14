import argparse
import os
from copy import deepcopy

import torch

from model_chroma_hnerv import convert_full_state_dict_to_native_rgbsplit


def verify_equivalence(source_state, converted_state, args):
    from model_all import HNeRV
    from model_chroma_hnerv import RGBSplitHNeRV
    from train_chroma_hnerv import build_parser, configure_model_size_args

    model_args = build_parser().parse_args([
        "--experiment", f"rgbsplit_{args.split_stage}", "--split_stage", args.split_stage,
        "--branch_width_mode", "native", "--modelsize", str(args.modelsize),
        "--enc_strds", "5", "4", "4", "2", "2", "--dec_strds", *map(str, args.dec_strds),
        "--enc_dim", "64_16", "--ks", "0_1_5", "--reduce", "1.2", "--lower_width", "12",
        "--conv_type", "convnext", "pshuffel", "--num_blks", args.num_blks,
    ])
    model_args.final_size = args.height * args.width
    model_args.full_data_length = args.frame_count
    configure_model_size_args(model_args)
    full, split = HNeRV(deepcopy(model_args)).eval(), RGBSplitHNeRV(deepcopy(model_args)).eval()
    normalized_source = {}
    for key, value in source_state.items():
        name = key[7:] if key.startswith("module.") else key
        if name.startswith("blocks.0."):
            name = name[len("blocks.0."):]
        normalized_source[name] = value
    full.load_state_dict(normalized_source, strict=True)
    split.load_state_dict(converted_state, strict=True)
    embed_channels = int(model_args.enc_dim.split("_")[1])
    embedding = torch.randn(1, embed_channels, args.height // 320, args.width // 320)
    with torch.no_grad():
        full_output = full(torch.zeros(1), input_embed=embedding)[0]
        split_output = split(torch.zeros(1), input_embed=embedding)[0]
    error = (full_output - split_output).abs().max().item()
    if error >= args.tolerance:
        raise RuntimeError(f"Converted model failed equivalence: max_abs_error={error}.")
    return error


def main():
    parser = argparse.ArgumentParser(description="Convert Full HNeRV weights to native/full-width RGBSplit.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modelsize", required=True, type=float)
    parser.add_argument("--split_stage", default="a160", choices=["a160", "a320"])
    parser.add_argument("--dec_strds", nargs="+", type=int, default=[5, 4, 4, 2, 2])
    parser.add_argument("--num_blks", default="1_1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--frame_count", type=int, default=132)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; pass --force to replace it.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source_state, dict):
        raise ValueError("Checkpoint does not contain a state_dict mapping.")
    converted = convert_full_state_dict_to_native_rgbsplit(
        source_state, args.dec_strds, args.num_blks, args.split_stage)
    max_abs_error = verify_equivalence(source_state, converted, args) if args.verify else None
    output = dict(checkpoint) if "state_dict" in checkpoint else {}
    output["state_dict"] = converted
    output["stage6_conversion"] = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_architecture": "rgb444_hnerv",
        "target_architecture": f"rgbsplit_{args.split_stage}",
        "branch_width_mode": "native",
        "modelsize": args.modelsize,
        "dec_strds": list(args.dec_strds),
        "num_blks": args.num_blks,
        "tensor_count": len(converted),
        "verified_max_abs_error": max_abs_error,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(output, args.output)
    print(f"Converted {args.checkpoint} -> {args.output}")
    print(f"Mapped tensors: {len(converted)}")
    if max_abs_error is not None:
        print(f"Verified max absolute output error: {max_abs_error:.3e}")


if __name__ == "__main__":
    main()
