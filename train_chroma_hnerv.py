import argparse
import csv
import io
import imageio
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from dahuffman import HuffmanCodec
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from hnerv_utils import *
from model_all import HNeRV, HNeRVDecoder, TransformInput, VideoDataSet, decoder_channel_schedule
from model_chroma_hnerv import ChromaHNeRV420, RGBSplitHNeRV
from uvg_utils import (
    apply_dataset_preset, atomic_torch_save, split_resolution_metadata,
)


RGB_STYLE_EXPERIMENTS = ("rgb444_hnerv", "rgbsplit_a320", "rgbsplit_a160")
CHROMA420_EXPERIMENTS = ("chroma420_a320", "chroma420_a160")
SPLIT_EXPERIMENTS = ("rgbsplit_a320", "chroma420_a320", "rgbsplit_a160", "chroma420_a160")
CHROMA_EXPERIMENTS = ("rgb444_hnerv", "ycbcr444_hnerv") + SPLIT_EXPERIMENTS
METRIC_COLUMNS = [
    "params_M", "encoder_params_M", "decoder_params_M", "embed_params_M",
    "actual_total_params_M", "modelsize_target_M", "checkpoint_size_MB",
    "estimated_gflops", "model_fps", "end_to_end_fps", "output_sample_ratio",
    "quant_model_bit", "quant_embed_bit", "bits_per_param",
    "bits_per_param_with_overhead", "bits_per_pixel",
    "rgb_psnr", "rgb_ms_ssim", "psnr_y", "psnr_cb", "psnr_cr",
    "yuv_psnr_611_dbavg", "yuv_psnr_611_mse",
    "ssim_y", "ssim_cb", "ssim_cr", "yuv_ssim_611", "lpips_alex",
    "frame_psnr_mean", "frame_psnr_std", "frame_y_psnr_mean",
    "frame_y_psnr_std", "temporal_rgb_error_diff",
]
SUPPORTED_QUANT_BITS = tuple(range(2, 9))
STORAGE_FIELDS = [
    "shared_param_count", "y_param_count", "chroma_param_count", "rgb_param_count",
    "model_param_count", "quantized_param_count", "shared_payload_bits", "y_payload_bits",
    "chroma_payload_bits", "rgb_payload_bits", "model_payload_bits", "embedding_payload_bits",
    "scale_min_overhead_bits", "buffer_bits", "metadata_bits", "tensor_padding_bits",
    "total_fixed_width_bits", "effective_bits_per_weight", "effective_bits_per_stored_value",
    "fixed_width_bpp", "huffman_payload_bits", "huffman_overhead_bits", "huffman_total_bits",
    "huffman_bpp", "packed_checkpoint_size_MB", "legacy_uint8_checkpoint_size_MB",
    "fixed_width_weight_bits", "fixed_width_total_bits", "packed_checkpoint_bytes",
    "effective_bits_per_playback_weight", "embedding_bits",
]


def build_parser():
    sanity_epilog = """
Sanity-check examples (small/debug only, not full experiments):
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment rgb444_hnerv --modelsize 0.35 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment ycbcr444_hnerv --modelsize 0.35 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment rgbsplit_a320 --modelsize 0.35 --branch_width 8 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment chroma420_a320 --modelsize 0.35 --branch_width 8 --lambda_rgb 0.1 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment rgbsplit_a160 --modelsize 0.35 --branch_width 8 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment chroma420_a160 --modelsize 0.35 --branch_width 8 --lambda_rgb 0.1 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
"""
    parser = argparse.ArgumentParser(
        description="Train RGB-HNeRV or full-resolution BT.709 YCbCr-HNeRV controls.",
        epilog=sanity_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default="", help="data path for vid")
    parser.add_argument("--vid", type=str, default="k400_train0", help="video id")
    parser.add_argument("--shuffle_data", action="store_true", help="randomly shuffle the frame idx")
    parser.add_argument("--data_split", type=str, default="1_1_1",
        help="Valid_train/total_train/all data split")
    parser.add_argument("--crop_list", type=str, default="640_1280", help="video crop size")
    parser.add_argument("--resize_list", type=str, default="-1", help="video resize size")
    parser.add_argument("--dataset_preset", choices=["none", "uvg_hnerv"], default="none")
    parser.add_argument("--expected_frames", type=int, default=-1)
    parser.add_argument("--expected_source_height", type=int, default=-1)
    parser.add_argument("--expected_source_width", type=int, default=-1)

    parser.add_argument("--embed", type=str, default="", help="empty for HNeRV; pe/le string for NeRV")
    parser.add_argument("--ks", type=str, default="0_3_3", help="kernel size for encoder and decoder")
    parser.add_argument("--enc_strds", type=int, nargs="+", default=[], help="stride list for encoder")
    parser.add_argument("--enc_dim", type=str, default="64_16", help="enc latent dim and embedding ratio")
    parser.add_argument("--modelsize", type=float, default=1.5, help="target model size in M params")
    parser.add_argument("--saturate_stages", type=int, default=-1, help="saturate stages for model size")
    parser.add_argument("--fc_hw", type=str, default="9_16", help="out size (h,w) for mlp")
    parser.add_argument("--reduce", type=float, default=1.2, help="channel reduction per stage")
    parser.add_argument("--lower_width", type=int, default=32, help="lowest channel width")
    parser.add_argument("--dec_strds", type=int, nargs="+", default=[5, 3, 2, 2, 2], help="decoder strides")
    parser.add_argument("--num_blks", type=str, default="1_1", help="encoder and decoder block counts")
    parser.add_argument("--conv_type", default=["convnext", "pshuffel"], type=str, nargs="+",
        help="conv type for encoder/decoder", choices=["pshuffel", "conv", "convnext", "interpolate"])
    parser.add_argument("--norm", default="none", type=str, help="norm layer", choices=["none", "bn", "in"])
    parser.add_argument("--act", type=str, default="gelu", help="activation",
        choices=["relu", "leaky", "leaky01", "relu6", "gelu", "swish", "softplus", "hardswish"])

    parser.add_argument("-j", "--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("-b", "--batchSize", type=int, default=1, help="input batch size")
    parser.add_argument("--start_epoch", type=int, default=-1, help="starting epoch")
    parser.add_argument("--not_resume", action="store_true", help="not resume from latest checkpoint")
    parser.add_argument("-e", "--epochs", type=int, default=150, help="Epoch number")
    parser.add_argument("--block_params", type=str, default="1_1", help="residual blocks and percentile")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_type", type=str, default="cosine_0.1_1_0.1", help="learning rate type")
    parser.add_argument("--loss", type=str, default="Fusion6", help="loss type")
    parser.add_argument("--out_bias", default="tanh", type=str, help="sigmoid/tanh/constant output bias")

    parser.add_argument("--eval_only", action="store_true", default=False, help="do evaluation only")
    parser.add_argument("--eval_freq", type=int, default=30, help="evaluation frequency")
    parser.add_argument("--quant_model_bit", type=int, default=8, help="bit length for model quantization")
    parser.add_argument("--quant_embed_bit", type=int, default=6, help="bit length for embedding quantization")
    parser.add_argument("--quant_axis", type=int, default=0, help="quantization axis (-1 means per tensor)")
    parser.add_argument("--quant_scheme", choices=["uniform", "component"], default="uniform")
    parser.add_argument("--quant_shared_bit", type=int, default=-1)
    parser.add_argument("--quant_y_bit", type=int, default=-1)
    parser.add_argument("--quant_chroma_bit", type=int, default=-1)
    parser.add_argument("--quant_rgb_bit", type=int, default=-1)
    parser.add_argument("--quant_tag", type=str, default="")
    parser.add_argument("--quant_matrix", type=str, default="")
    parser.add_argument("--rd_curve", type=str, default="")
    parser.add_argument("--save_packed_quant", action="store_true")
    parser.add_argument("--quant_storage_mode", choices=["packed", "uint8_legacy"], default="packed")
    parser.add_argument("--dump_images", action="store_true", default=False, help="dump prediction images")
    parser.add_argument("--dump_videos", action="store_true", default=False, help="concat predictions into video")
    parser.add_argument("--eval_fps", action="store_true", default=False, help="forward multiple times for fps")
    parser.add_argument("--measure_latency", action="store_true", default=False)
    parser.add_argument("--intermediate_eval_mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--final_eval_mode", choices=["quick", "full"], default="full")
    parser.add_argument("--encoder_file", default="", type=str, help="specify the embedding file")

    parser.add_argument("--manualSeed", type=int, default=1, help="manual seed")
    parser.add_argument("-d", "--distributed", action="store_true", default=False, help="distributed training")
    parser.add_argument("--debug", action="store_true", help="debug status, early train/eval")
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    parser.add_argument("--weight", default="None", type=str, help="pretrained weights")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the output dir")
    parser.add_argument("--outf", default="unify", help="folder to output images and model checkpoints")
    parser.add_argument("--run_dir", default="", help="exact run directory; primarily for suite launchers")
    parser.add_argument("--suffix", default="", help="suffix str for outf")
    parser.add_argument("--backup_root", default="")
    parser.add_argument("--backup_at_eval", action="store_true")
    parser.add_argument("--strict_backup", action="store_true")
    parser.add_argument("--launcher_log_path", default="")

    parser.add_argument("--experiment", default="rgb444_hnerv", choices=CHROMA_EXPERIMENTS)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--results_csv", default="", type=str)
    parser.add_argument("--lambda_y", default=1.0, type=float)
    parser.add_argument("--lambda_c", default=1.0, type=float)
    parser.add_argument("--lambda_rgb", default=0.0, type=float)
    parser.add_argument("--split_stage", default="a320", choices=["a320", "a160"])
    parser.add_argument("--branch_width", default=8, type=int)
    parser.add_argument("--branch_width_mode", default="fixed", choices=["fixed", "native"])
    parser.add_argument("--chroma_scale", default=2, type=int)
    parser.add_argument("--chroma_downsample", default="area", choices=["area"])
    parser.add_argument("--chroma_upsample", default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument("--profile_only", action="store_true")
    return parser


def main():
    parser = build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    apply_dataset_preset(args, parser, argv)
    normalize_experiment_args(args)
    torch.set_printoptions(precision=4)
    setup_output_dir(args)
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method = f"tcp://127.0.0.1:{port}"
    print(f"init_method: {args.init_method}", flush=True)

    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)


def normalize_experiment_args(args):
    lambda_rgb_was_explicit = any(arg == "--lambda_rgb" or arg.startswith("--lambda_rgb=") for arg in sys.argv[1:])
    if args.experiment in CHROMA420_EXPERIMENTS and not lambda_rgb_was_explicit:
        args.lambda_rgb = 0.1
    branch_width_explicit = any(arg == "--branch_width" or arg.startswith("--branch_width=") for arg in sys.argv[1:])
    if args.branch_width_mode == "native" and branch_width_explicit:
        raise ValueError("--branch_width must not be supplied with --branch_width_mode native.")
    if args.branch_width_mode == "native" and args.experiment not in SPLIT_EXPERIMENTS:
        raise ValueError("--branch_width_mode native is only valid for RGBSplit and Chroma420 models.")
    validate_quant_args(args)

    expected_split = None
    if args.experiment.endswith("_a320"):
        expected_split = "a320"
    elif args.experiment.endswith("_a160"):
        expected_split = "a160"
    if expected_split is None:
        return

    split_arg_was_explicit = any(arg == "--split_stage" or arg.startswith("--split_stage=") for arg in sys.argv[1:])
    if split_arg_was_explicit and args.split_stage != expected_split:
        raise ValueError(
            f"{args.experiment} requires --split_stage {expected_split}, "
            f"but got --split_stage {args.split_stage}."
        )
    args.split_stage = expected_split


def validate_quant_args(args):
    if args.quant_axis < -2:
        raise ValueError("--quant_axis must be -2 (automatic), -1 (per tensor), or a tensor axis.")
    for name in ["quant_model_bit", "quant_embed_bit", "quant_shared_bit", "quant_y_bit",
                 "quant_chroma_bit", "quant_rgb_bit"]:
        value = getattr(args, name)
        if value != -1 and value not in SUPPORTED_QUANT_BITS:
            raise ValueError(f"--{name} must be -1 or one of {SUPPORTED_QUANT_BITS}, got {value}.")
    if args.quant_embed_bit == -1 and (args.quant_scheme == "component" or args.quant_model_bit != -1):
        raise ValueError("--quant_embed_bit cannot be -1 when model PTQ is enabled.")
    if args.quant_scheme == "uniform":
        return
    if args.experiment in ("rgb444_hnerv", "ycbcr444_hnerv"):
        raise ValueError("Component quantization is unsupported for full HNeRV; use --quant_scheme uniform.")
    required = ["quant_shared_bit"]
    required += ["quant_rgb_bit"] if args.experiment.startswith("rgbsplit_") else ["quant_y_bit", "quant_chroma_bit"]
    for name in required:
        value = getattr(args, name)
        if value not in SUPPORTED_QUANT_BITS:
            raise ValueError(f"Component mode requires --{name} in {SUPPORTED_QUANT_BITS}, got {value}.")


def setup_output_dir(args):
    if args.debug:
        args.eval_freq = 1
    if not args.run_dir and args.debug:
        args.outf = "output/debug"
    elif not args.run_dir:
        args.outf = os.path.join("output", args.outf)
    args.enc_strd_str = ",".join([str(x) for x in args.enc_strds])
    args.dec_strd_str = ",".join([str(x) for x in args.dec_strds])
    extra_str = "Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}".format(
        args.modelsize, args.conv_type[0], args.enc_strd_str, args.conv_type[1],
        args.dec_strd_str, "" if args.norm == "none" else f"_{args.norm}",
        "_dist" if args.distributed else "", "_shuffle_data" if args.shuffle_data else "")
    if args.quant_scheme == "uniform":
        args.quant_str = f"uniform_M{args.quant_model_bit}_E{args.quant_embed_bit}"
    elif args.experiment.startswith("rgbsplit_"):
        args.quant_str = f"component_S{args.quant_shared_bit}_RGB{args.quant_rgb_bit}_E{args.quant_embed_bit}"
    else:
        args.quant_str = (
            f"component_S{args.quant_shared_bit}_Y{args.quant_y_bit}"
            f"_C{args.quant_chroma_bit}_E{args.quant_embed_bit}")
    if args.quant_tag:
        args.quant_str += f"_{args.quant_tag}"
    embed_str = f"{args.embed}_Dim{args.enc_dim}"
    run_prefix = f"{args.experiment}_" + (f"{args.run_name}_" if args.run_name else "")
    if args.experiment in SPLIT_EXPERIMENTS:
        split_str = (f"_split{args.split_stage}_native" if args.branch_width_mode == "native"
                     else f"_split{args.split_stage}_w{args.branch_width}")
    else:
        split_str = ""
    args.exp_id = run_prefix + (
        f"{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}"
        f"_low{args.lower_width}_blk{args.num_blks}_e{args.epochs}_b{args.batchSize}_{args.quant_str}"
        f"_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{split_str}{args.suffix}"
    )
    args.run_relative_path = os.path.basename(os.path.normpath(args.run_dir)) if args.run_dir else args.exp_id
    args.outf = args.run_dir if args.run_dir else os.path.join(args.outf, args.exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print("Will overwrite the existing output dir!")
        shutil.rmtree(args.outf)
    os.makedirs(args.outf, exist_ok=True)
    with open(os.path.join(args.outf, "command.txt"), "w", encoding="utf-8") as file:
        file.write(shlex.join([sys.executable, *sys.argv]) + "\n")
    split_alias = args.split_stage if args.experiment in SPLIT_EXPERIMENTS else None
    for key, value in split_resolution_metadata(args.crop_list, args.dec_strds, split_alias).items():
        setattr(args, key, value)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unavailable", False


def atomic_json_dump(payload, destination):
    temporary = destination + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def write_run_metadata(args):
    import torchvision

    commit, dirty = git_state()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    metadata = {key: _json_safe(value) for key, value in vars(args).items()
                if key not in {"transform_func"}}
    metadata.update({
        "sequence_name": args.vid,
        "dataset_path": args.data_path,
        "timestamp": datetime.now().astimezone().isoformat(),
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
    })
    atomic_json_dump(metadata, os.path.join(args.outf, "config.json"))
    with open(os.path.join(args.outf, "git_commit.txt"), "w", encoding="utf-8") as file:
        file.write(f"commit={commit}\ndirty={str(dirty).lower()}\n")

    environment_path = os.path.join(args.outf, "environment.txt")
    if not os.path.exists(environment_path):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, check=True)
            environment = result.stdout
        except Exception as exc:
            environment = f"pip freeze unavailable: {exc}\n"
        with open(environment_path, "w", encoding="utf-8") as file:
            file.write(environment)


def backup_run_artifacts(args, final=False):
    if not args.backup_root:
        return
    destination = os.path.join(args.backup_root, args.run_relative_path)
    names = ["model_latest.pth", "rank0.txt", "config.json", "command.txt"]
    if args.launcher_log_path:
        names.append(args.launcher_log_path)
    csv_files = sorted(
        [name for name in os.listdir(args.outf) if name.endswith(".csv")],
        key=lambda name: os.path.getmtime(os.path.join(args.outf, name)))
    if csv_files:
        names.append(csv_files[-1])
    if final:
        names.extend(["model_best.pth", f"epoch{args.epochs}.pth",
                      f"epoch{args.epochs}.csv", "completion.json"])
    try:
        os.makedirs(destination, exist_ok=True)
        for name in dict.fromkeys(names):
            source = name if os.path.isabs(name) else os.path.join(args.outf, name)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(destination, os.path.basename(source)))
    except Exception as exc:
        message = f"Backup failed for {args.outf}: {exc}"
        print(message, flush=True)
        if args.strict_backup:
            raise RuntimeError(message) from exc


def data_to_gpu(x, device):
    return x.to(device)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_chroma420_experiment(args):
    return args.experiment in CHROMA420_EXPERIMENTS


def is_split_experiment(args):
    return args.experiment in SPLIT_EXPERIMENTS


def quantization_enabled(args):
    return args.quant_scheme == "component" or args.quant_model_bit != -1


def architecture_metadata(args):
    if args.experiment in ["rgb444_hnerv", "ycbcr444_hnerv"]:
        return {
            "architecture": "hnerv_444",
            "chroma_format": "444",
            "split_stage": "none",
            "branch_width": float("nan"),
            "branch_width_mode": "none",
            "chroma_scale": 1,
            "output_sample_ratio": 1.0,
        }
    if args.experiment.startswith("rgbsplit_"):
        return {
            "architecture": args.experiment,
            "chroma_format": "rgb444",
            "split_stage": args.split_stage,
            "branch_width": float("nan") if args.branch_width_mode == "native" else args.branch_width,
            "branch_width_mode": args.branch_width_mode,
            "chroma_scale": 1,
            "output_sample_ratio": 1.0,
        }
    if is_chroma420_experiment(args):
        return {
            "architecture": args.experiment,
            "chroma_format": "420",
            "split_stage": args.split_stage,
            "branch_width": float("nan") if args.branch_width_mode == "native" else args.branch_width,
            "branch_width_mode": args.branch_width_mode,
            "chroma_scale": 2,
            "output_sample_ratio": 0.5,
        }
    raise ValueError(f"Unsupported experiment: {args.experiment}")


def build_model(args):
    if args.experiment in ["rgb444_hnerv", "ycbcr444_hnerv"]:
        return HNeRV(args)
    if args.experiment.startswith("rgbsplit_"):
        return RGBSplitHNeRV(args)
    if is_chroma420_experiment(args):
        return ChromaHNeRV420(args)
    raise ValueError(f"Unsupported experiment: {args.experiment}")


def configure_model_size_args(args):
    if "pe" in args.embed or "le" in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split("_")[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split("_")])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds ** 2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split("_")]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio)
        embed_param = float(embed_dim) / total_enc_strds ** 2 * args.final_size * args.full_data_length
        args.enc_dim = f"{int(enc_dim1)}_{embed_dim}"
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds)) ** 2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1.0 / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split("_")[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a = ch_reduce * sum([
        ch_reduce ** (2 * i) * s ** 2 * min((2 * i + dec_ks1), dec_ks2) ** 2
        for i, s in enumerate(args.dec_strds[:fix_ch_stages])
    ])
    b = embed_dim * fc_param
    c = args.lower_width ** 2 * sum([
        s ** 2 * min(2 * (fix_ch_stages + i) + dec_ks1, dec_ks2) ** 2
        for i, s in enumerate(args.dec_strds[fix_ch_stages:])
    ])
    args.fc_dim = int(np.roots([a, b, c - decoder_size]).max())
    args.embed_params_M = embed_param / 1e6


def attach_param_counts(args, model):
    base = unwrap_model(model)
    encoder_params = 0
    decoder_params = 0
    for name, param in base.named_parameters():
        if "encoder" in name:
            encoder_params += param.data.nelement()
        else:
            decoder_params += param.data.nelement()
    args.encoder_params_M = encoder_params / 1e6
    args.decoder_params_M = decoder_params / 1e6
    args.actual_total_params_M = (encoder_params + decoder_params) / 1e6 + args.embed_params_M
    args.params_M = args.decoder_params_M + args.embed_params_M
    args.encoder_param = args.encoder_params_M
    args.decoder_param = args.decoder_params_M
    args.total_param = args.params_M
    args.actual_total_params = encoder_params + decoder_params + int(round(args.embed_params_M * 1e6))
    args.decoder_playback_params = decoder_params

    groups = {"shared": 0, "rgb": 0, "y": 0, "chroma": 0, "model": 0}
    for name, param in base.named_parameters():
        if name.startswith("shared_decoder."):
            groups["shared"] += param.numel()
        elif name.startswith(("rgb_branch.", "rgb_head.")):
            groups["rgb"] += param.numel()
        elif name.startswith(("y_branch.", "y_head.")):
            groups["y"] += param.numel()
        elif name.startswith(("cbcr_branch.", "cbcr_head.")):
            groups["chroma"] += param.numel()
        elif name.startswith(("decoder.", "head_layer.")):
            groups["model"] += param.numel()
    for group, count in groups.items():
        setattr(args, f"architecture_{group}_param_count", count)
        setattr(args, f"architecture_{group}_param_percent", 100 * count / max(decoder_params, 1))

    native_schedule = tuple(decoder_channel_schedule(args))
    args.native_channel_schedule = ",".join(map(str, native_schedule))
    args.shared_channels = ",".join(map(str, getattr(base, "shared_channel_schedule", ())))
    args.rgb_branch_channels = ",".join(map(str, getattr(base, "rgb_branch_channels", ())))
    args.y_branch_channels = ",".join(map(str, getattr(base, "y_branch_channels", ())))
    args.chroma_branch_channels = ",".join(map(str, getattr(base, "chroma_branch_channels", ())))


def interpret_prediction(raw_output, args, aux=None):
    if args.experiment in RGB_STYLE_EXPERIMENTS:
        rgb = raw_output
        ycbcr = rgb_to_ycbcr_bt709(rgb)
    elif args.experiment == "ycbcr444_hnerv":
        ycbcr = raw_output
        rgb = ycbcr_to_rgb_bt709(ycbcr)
    elif is_chroma420_experiment(args):
        rgb = aux["rgb"] if aux is not None and "rgb" in aux else raw_output
        ycbcr = aux["ycbcr"] if aux is not None and "ycbcr" in aux else rgb_to_ycbcr_bt709(rgb)
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")
    pred = {"rgb": rgb, "ycbcr": ycbcr, "raw": raw_output}
    if aux is not None:
        for key in ["y", "cbcr_low", "cbcr_up"]:
            if key in aux:
                pred[key] = aux[key]
    return pred


def area_downsample_chroma(chroma, scale):
    return F.interpolate(chroma, scale_factor=1.0 / scale, mode="area")


def compute_train_loss(raw_output, img_gt, inpaint_mask, args, aux=None):
    if args.experiment in RGB_STYLE_EXPERIMENTS:
        loss = loss_fn(raw_output * inpaint_mask, img_gt * inpaint_mask, args.loss)
        return loss, {"loss": loss.detach(), "loss_rgb": loss.detach()}

    if args.experiment == "ycbcr444_hnerv" and "inpaint" in args.vid:
        raise NotImplementedError("YCbCr-HNeRV inpainting is not implemented in this first experiment.")

    if args.experiment == "ycbcr444_hnerv":
        pred_ycbcr = raw_output
        target_ycbcr = rgb_to_ycbcr_bt709(img_gt)
        loss_y = F.mse_loss(pred_ycbcr[:, 0:1], target_ycbcr[:, 0:1])
        loss_cb = F.mse_loss(pred_ycbcr[:, 1:2], target_ycbcr[:, 1:2])
        loss_cr = F.mse_loss(pred_ycbcr[:, 2:3], target_ycbcr[:, 2:3])
        loss_c = 0.5 * (loss_cb + loss_cr)
        pred_rgb = ycbcr_to_rgb_bt709(pred_ycbcr)
        loss_rgb = F.mse_loss(pred_rgb, img_gt)
        loss = args.lambda_y * loss_y + args.lambda_c * loss_c + args.lambda_rgb * loss_rgb
        return loss, {
            "loss": loss.detach(), "loss_y": loss_y.detach(), "loss_cb": loss_cb.detach(),
            "loss_cr": loss_cr.detach(), "loss_c": loss_c.detach(), "loss_rgb": loss_rgb.detach(),
        }

    if is_chroma420_experiment(args):
        if "inpaint" in args.vid:
            raise NotImplementedError("Chroma420-HNeRV inpainting is not implemented in this first experiment.")
        if aux is None:
            raise ValueError(f"{args.experiment} training requires auxiliary model outputs.")
        target_ycbcr = rgb_to_ycbcr_bt709(img_gt)
        target_y = target_ycbcr[:, 0:1]
        target_cbcr_low = area_downsample_chroma(target_ycbcr[:, 1:3], args.chroma_scale)
        loss_y = F.mse_loss(aux["y"], target_y)
        loss_c = F.mse_loss(aux["cbcr_low"], target_cbcr_low)
        loss_rgb = F.mse_loss(aux["rgb"], img_gt)
        loss = args.lambda_y * loss_y + args.lambda_c * loss_c + args.lambda_rgb * loss_rgb
        return loss, {
            "loss": loss.detach(), "loss_y": loss_y.detach(), "loss_c": loss_c.detach(),
            "loss_rgb": loss_rgb.detach(),
        }

    raise ValueError(f"Unsupported experiment: {args.experiment}")


def build_lpips_model(device):
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"LPIPS unavailable, logging nan for lpips_alex ({exc})", flush=True)
        return None


def safe_mean(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def tensor_min_max(tensor):
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return finite.min().item(), finite.max().item()


def assert_finite_tensor(tensor, name, context):
    if torch.isfinite(tensor).all():
        return
    min_v, max_v = tensor_min_max(tensor.detach())
    finite_count = torch.isfinite(tensor).sum().item()
    total_count = tensor.numel()
    raise FloatingPointError(
        f"Non-finite tensor detected in {context}: {name}, "
        f"finite={finite_count}/{total_count}, finite_min={min_v}, finite_max={max_v}"
    )


def safe_msssim(pred, target):
    try:
        return ms_ssim(pred.float().detach(), target.detach(), data_range=1, size_average=False).cpu()
    except Exception:
        return torch.full((pred.size(0),), float("nan"))


def safe_ssim(pred, target):
    try:
        return ssim(pred.float().detach(), target.detach(), data_range=1, size_average=False).cpu()
    except Exception:
        return torch.full((pred.size(0),), float("nan"))


@torch.no_grad()
def estimate_gflops(model, full_dataloader, args, device):
    base = unwrap_model(model)
    was_training = base.training
    base.eval()
    flops = {"total": 0}
    handles = []

    def conv_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.dim() != 4:
            return
        batch, out_c, out_h, out_w = out.shape
        kh, kw = module.kernel_size
        kernel_ops = kh * kw * (module.in_channels / module.groups)
        flops["total"] += batch * out_c * out_h * out_w * kernel_ops * 2

    def linear_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        flops["total"] += out.numel() * module.in_features * 2

    for module in base.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    try:
        sample = next(iter(full_dataloader))
        img_data = data_to_gpu(sample["img"], device)
        norm_idx = data_to_gpu(sample["norm_idx"], device)
        img_data, _, _ = args.transform_func(img_data)
        cur_input = norm_idx if "pe" in args.embed else img_data
        base(cur_input)
        return flops["total"] / max(cur_input.size(0), 1) / 1e9
    except Exception as exc:
        print(f"GFLOP estimate unavailable, logging nan ({exc})", flush=True)
        return float("nan")
    finally:
        for handle in handles:
            handle.remove()
        base.train(was_training)


def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.init_method,
            world_size=args.ngpus_per_node, rank=local_rank)
        torch.cuda.set_device(local_rank)
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    full_dataset = VideoDataSet(args)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    full_dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=False,
        worker_init_fn=worker_init_fn)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split("_")]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    args.dump_vis = args.dump_images or args.dump_videos

    train_dataset = Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=worker_init_fn)

    configure_model_size_args(args)
    model = build_model(args)
    attach_param_counts(args, model)

    if local_rank in [0, None]:
        param_str = (
            f"Encoder_{round(args.encoder_params_M, 2)}M_Decoder_{round(args.decoder_params_M, 2)}M_"
            f"Embed_{round(args.embed_params_M, 2)}M_Repr_{round(args.params_M, 2)}M"
        )
        print(f"{args}\n {model}\n {param_str}", flush=True)
        with open(f"{args.outf}/rank0.txt", "a") as f:
            f.write(str(model) + "\n" + f"{param_str}\n")
        writer = SummaryWriter(os.path.join(args.outf, param_str, "tensorboard"))
    else:
        writer = None

    print(f"Use GPU: {local_rank} for training")
    if args.distributed and args.ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.0)
    args.transform_func = TransformInput(args)
    device = next(model.parameters()).device
    if local_rank in [0, None]:
        args.estimated_gflops = estimate_gflops(model, full_dataloader, args, device)
        write_run_metadata(args)
    else:
        args.estimated_gflops = float("nan")

    if args.profile_only:
        if local_rank in [0, None]:
            args.cur_epoch, args.train_time = 0, 0
            metrics_by_variant = {"orig": build_profile_metrics(args)}
            dump_csv(args, metrics_by_variant, "profile.csv")
            append_consolidated_csv(args, metrics_by_variant)
            print("Profile-only mode complete; no training was run.", flush=True)
        return

    checkpoint = load_checkpoint_if_needed(model, optimizer, args, local_rank)
    if args.start_epoch < 0:
        args.start_epoch = checkpoint["epoch"] if checkpoint is not None else 0
        args.start_epoch = max(args.start_epoch, 0)

    if args.eval_only:
        metrics_by_variant, hw = evaluate(
            model, full_dataloader, local_rank, args, args.dump_vis,
            huffman_coding=True, evaluation_mode=args.final_eval_mode)
        if local_rank in [0, None]:
            args.train_time, args.cur_epoch = 0, args.epochs
            dump_csv(args, metrics_by_variant, "eval.csv")
            append_consolidated_csv(args, metrics_by_variant)
            print(f"Evaluation complete for {hw}", flush=True)
        return

    start = datetime.now()
    best_rgb_psnr = -float("inf")
    last_metrics = None
    for epoch in range(args.start_epoch, args.epochs):
        evaluated_this_epoch = False
        is_best_epoch = False
        model.train()
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        for i, sample in enumerate(train_dataloader):
            img_data = data_to_gpu(sample["img"], device)
            norm_idx = data_to_gpu(sample["norm_idx"], device)
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            cur_input = norm_idx if "pe" in args.embed else img_data
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            if is_chroma420_experiment(args):
                raw_output, _, _, aux = model(cur_input, return_aux=True)
            else:
                raw_output, _, _ = model(cur_input)
                aux = None
            final_loss, loss_log = compute_train_loss(raw_output, img_gt, inpaint_mask, args, aux)
            pred_rgb = interpret_prediction(raw_output.detach(), args, aux)["rgb"]

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            pred_psnr_list.append(psnr_fn_single(pred_rgb.detach(), img_gt))

            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                loss_bits = " ".join([f"{k}:{v.item():.4f}" for k, v in loss_log.items()])
                print_str = (
                    f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Rank:{local_rank}, "
                    f"Epoch[{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                    f"lr:{lr:.2e} pred_PSNR:{RoundTensor(pred_psnr, 2)} {loss_bits}"
                )
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open(f"{args.outf}/rank0.txt", "a") as f:
                        f.write(print_str + "\n")

        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        if local_rank in [0, None]:
            h, w = pred_rgb.shape[-2:]
            writer.add_scalar(f"Train/pred_PSNR_{h}X{w}", pred_psnr, epoch + 1)
            writer.add_scalar("Train/lr", lr, epoch + 1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
                (epoch_end_time - epoch_start_time).total_seconds(),
                (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch)))

        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            is_final_epoch = epoch + 1 == args.epochs
            evaluation_mode = args.final_eval_mode if is_final_epoch else args.intermediate_eval_mode
            last_metrics, hw = evaluate(
                model, full_dataloader, local_rank, args,
                args.dump_vis if is_final_epoch and evaluation_mode == "full" else False,
                huffman_coding=(is_final_epoch and evaluation_mode == "full"),
                evaluation_mode=evaluation_mode)
            evaluated_this_epoch = True
            if local_rank in [0, None]:
                cur_psnr = last_metrics["orig"].get("rgb_psnr", float("nan"))
                is_best_epoch = cur_psnr >= best_rgb_psnr
                best_rgb_psnr = max(best_rgb_psnr, cur_psnr)
                writer.add_scalar(f"Val/rgb_psnr_{hw}", cur_psnr, epoch + 1)
                print_str = f"Eval at epoch {epoch + 1} for {hw}: rgb_psnr: {cur_psnr:.2f} best_rgb_psnr: {best_rgb_psnr:.2f}"
                print(print_str, flush=True)
                with open(f"{args.outf}/rank0.txt", "a") as f:
                    f.write(print_str + "\n")

        save_checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture_config": {
                "experiment": args.experiment,
                "modelsize": args.modelsize,
                "split_stage": args.split_stage if args.experiment in SPLIT_EXPERIMENTS else None,
                "branch_width_mode": args.branch_width_mode if args.experiment in SPLIT_EXPERIMENTS else None,
                "branch_width": (args.branch_width if args.experiment in SPLIT_EXPERIMENTS
                                 and args.branch_width_mode == "fixed" else None),
                "fc_dim": args.fc_dim,
                "dec_strds": list(args.dec_strds),
                "native_channel_schedule": getattr(args, "native_channel_schedule", ""),
                "shared_channels": getattr(args, "shared_channels", ""),
                "rgb_branch_channels": getattr(args, "rgb_branch_channels", ""),
                "y_branch_channels": getattr(args, "y_branch_channels", ""),
                "chroma_branch_channels": getattr(args, "chroma_branch_channels", ""),
                "decoder_stage_resolutions": args.decoder_stage_resolutions,
                "split_alias": args.split_alias,
                "split_stage_index": args.split_stage_index,
                "shared_output_height": args.shared_output_height,
                "shared_output_width": args.shared_output_width,
                "chroma_output_height": args.chroma_output_height,
                "chroma_output_width": args.chroma_output_width,
                "final_output_height": args.final_output_height,
                "final_output_width": args.final_output_width,
            },
        }
        if local_rank in [0, None]:
            atomic_torch_save(save_checkpoint, f"{args.outf}/model_latest.pth")
            if evaluated_this_epoch and is_best_epoch:
                atomic_torch_save(save_checkpoint, f"{args.outf}/model_best.pth")
            if evaluated_this_epoch and args.backup_at_eval:
                backup_run_artifacts(args)
            if epoch + 1 == args.epochs:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                if last_metrics is None:
                    last_metrics, _ = evaluate(
                        model, full_dataloader, local_rank, args, args.dump_vis,
                        huffman_coding=(args.final_eval_mode == "full"),
                        evaluation_mode=args.final_eval_mode)
                dump_csv(args, last_metrics, f"epoch{epoch + 1}.csv")
                append_consolidated_csv(args, last_metrics)
                atomic_torch_save(save_checkpoint, f"{args.outf}/epoch{epoch + 1}.pth")
                if not os.path.isfile(f"{args.outf}/model_best.pth"):
                    atomic_torch_save(save_checkpoint, f"{args.outf}/model_best.pth")
                completion = {
                    "status": "complete",
                    "epoch": epoch + 1,
                    "run_name": args.run_name,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "final_evaluation_mode": args.final_eval_mode,
                    "quantization": f"M{args.quant_model_bit}/E{args.quant_embed_bit}",
                }
                atomic_json_dump(completion, os.path.join(args.outf, "completion.json"))
                write_run_metadata(args)
                backup_run_artifacts(args, final=True)

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


def load_checkpoint_if_needed(model, optimizer, args, local_rank):
    checkpoint = None
    if args.weight != "None":
        print(f"=> loading checkpoint '{args.weight}'")
        checkpoint = torch.load(args.weight, map_location="cpu")
        orig_ckt = checkpoint["state_dict"]
        new_ckt = {}
        for key, value in orig_ckt.items():
            name = key[7:] if key.startswith("module.") else key
            if name.startswith("blocks.0."):
                name = name[len("blocks.0."):]
            new_ckt[name] = value
        unwrap_model(model).load_state_dict(new_ckt, strict=True)
        print(f"=> loaded checkpoint '{args.weight}' (epoch {checkpoint['epoch']})")

    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, "model_latest.pth")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> Auto resume loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No resume checkpoint found at '{checkpoint_path}'")
    return checkpoint


@torch.no_grad()
def evaluate(
        model, full_dataloader, local_rank, args, dump_vis=False,
        huffman_coding=False, evaluation_mode="full"):
    full_evaluation = evaluation_mode == "full"
    img_embed_list = []
    if full_evaluation:
        model_list, quant_ckt, quant_report = quant_model(model, args)
    else:
        model_list, quant_ckt, quant_report = [deepcopy(model)], None, None
        dump_vis = False
        huffman_coding = False
    metrics_by_variant = {}
    dequant_vid_embed = None
    quant_embed = None
    lpips_model = build_lpips_model(next(model.parameters()).device) if full_evaluation else None

    for model_ind, cur_model in enumerate(model_list):
        variant = "quant" if model_ind else "orig"
        cur_model.eval()
        device = next(cur_model.parameters()).device
        metric_lists = {name: [] for name in METRIC_COLUMNS}
        frame_psnr, frame_y_psnr, temporal_diffs, lpips_vals = [], [], [], []
        prev_idx, prev_err = None, None
        fwd_time, end_time, frame_count = 0.0, 0.0, 0

        if dump_vis:
            visual_dir = f"{args.outf}/visualize_model_{variant}"
            print(f"Saving predictions to {visual_dir}...")
            os.makedirs(visual_dir, exist_ok=True)

        for i, sample in enumerate(full_dataloader):
            step_start = time.time()
            img_data = data_to_gpu(sample["img"], device)
            norm_idx = data_to_gpu(sample["norm_idx"], device)
            img_idx = data_to_gpu(sample["idx"], device)
            if i > 10 and args.debug:
                break
            img_data, img_gt, _ = args.transform_func(img_data)
            cur_input = norm_idx if "pe" in args.embed else img_data

            fwd_start = time.time()
            input_embed = dequant_vid_embed[i] if model_ind else None
            if is_chroma420_experiment(args):
                raw_output, embed_list, dec_time, aux = cur_model(cur_input, input_embed, return_aux=True)
            else:
                raw_output, embed_list, dec_time = cur_model(cur_input, input_embed)
                aux = None
            if (args.measure_latency or (full_evaluation and args.eval_fps)) and torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_time += time.time() - fwd_start

            if full_evaluation and model_ind == 0:
                img_embed_list.append(embed_list[0])
            if full_evaluation and args.eval_fps:
                for _ in range(100):
                    if is_chroma420_experiment(args):
                        raw_output, embed_list, _, aux = cur_model(cur_input, embed_list[0], return_aux=True)
                    else:
                        raw_output, embed_list, _ = cur_model(cur_input, embed_list[0])

            pred = interpret_prediction(raw_output, args, aux)
            pred_rgb = pred["rgb"].clamp(0, 1)
            pred_ycbcr = pred["ycbcr"].clamp(0, 1)
            finite_context = f"experiment={args.experiment}, variant={variant}, batch={i}"
            assert_finite_tensor(pred_rgb, "pred_rgb", finite_context)
            assert_finite_tensor(pred_ycbcr, "pred_ycbcr", finite_context)
            gt_rgb = img_gt.clamp(0, 1)
            gt_ycbcr = rgb_to_ycbcr_bt709(gt_rgb).clamp(0, 1)
            end_time += time.time() - step_start
            frame_count += gt_rgb.size(0)

            batch_metrics = (
                compute_quality_metrics(pred_rgb, gt_rgb, pred_ycbcr, gt_ycbcr)
                if full_evaluation else compute_quick_quality_metrics(
                    pred_rgb, gt_rgb, pred_ycbcr, gt_ycbcr))
            for key, values in batch_metrics.items():
                metric_lists[key].extend(values)
            frame_psnr.extend(batch_metrics["frame_psnr_mean"])
            frame_y_psnr.extend(batch_metrics["frame_y_psnr_mean"])

            if full_evaluation and lpips_model is not None:
                try:
                    lp = lpips_model(pred_rgb * 2 - 1, gt_rgb * 2 - 1).flatten().detach().cpu().tolist()
                    lpips_vals.extend(lp)
                except Exception as exc:
                    print(f"LPIPS evaluation failed at batch {i}: {exc}", flush=True)
                    lpips_model = None

            if full_evaluation:
                for batch_ind, cur_img_idx in enumerate(img_idx.detach().cpu().tolist()):
                    err = (pred_rgb[batch_ind] - gt_rgb[batch_ind]).detach().cpu()
                    if prev_err is not None and cur_img_idx == prev_idx + 1:
                        temporal_diffs.append(torch.mean(torch.abs(err - prev_err)).item())
                    prev_idx, prev_err = cur_img_idx, err

            if dump_vis:
                for batch_ind, _ in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    temp_psnr = batch_metrics["frame_psnr_mean"][batch_ind]
                    concat_img = torch.cat([img_data[batch_ind], pred_rgb[batch_ind]], dim=2)
                    save_image(concat_img, f"{visual_dir}/pred_{full_ind:04d}_{temp_psnr:.2f}.png")

            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                timing = (
                    f", FPS {round(frame_count / max(fwd_time, 1e-12), 1)}"
                    if full_evaluation and (args.measure_latency or args.eval_fps) else "")
                print_str = (
                    f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Rank:{local_rank}, "
                    f"Eval {variant} Step [{i + 1}/{len(full_dataloader)}]{timing}, "
                    f"rgb_psnr: {safe_mean(metric_lists['rgb_psnr']):.2f}"
                )
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open(f"{args.outf}/rank0.txt", "a") as f:
                        f.write(print_str + "\n")

        metrics = summarize_metrics(args, metric_lists, frame_psnr, frame_y_psnr, temporal_diffs, lpips_vals)
        timing_enabled = full_evaluation and (args.measure_latency or args.eval_fps)
        metrics["model_fps"] = (
            frame_count / max(fwd_time, 1e-12) if frame_count and timing_enabled else float("nan"))
        metrics["end_to_end_fps"] = (
            frame_count / max(end_time, 1e-12) if frame_count and timing_enabled else float("nan"))
        metrics_by_variant[variant] = metrics

        if full_evaluation and model_ind == 0 and quantization_enabled(args):
            vid_embed = torch.cat(img_embed_list, 0)
            quant_embed, dequant_embed = quant_tensor(vid_embed, args.quant_embed_bit, args.quant_axis)
            dequant_vid_embed = dequant_embed.split(args.batchSize, dim=0)

        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, f"gt_pred_{variant}.gif")
            with imageio.get_writer(gif_file, mode="I") as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)

    if local_rank in [0, None] and quant_ckt is not None and quant_embed is not None:
        finalize_storage_report(args, quant_report, quant_embed, quant_ckt, huffman_coding)
        packed_embed = serialize_quant_tensor(quant_embed)
        packed_model = {key: serialize_quant_tensor(value) for key, value in quant_ckt.items()}
        common = {
            "format_version": 2,
            "quant_config": quant_config(args),
            "buffers": quant_report["buffers"],
            "storage_report": quant_report["storage"],
        }
        packed_vid = dict(common, embed=packed_embed, model=packed_model)
        legacy_vid = dict(common, embed=quant_embed, model=quant_ckt)
        for _ in range(3):
            packed_stream, legacy_stream = io.BytesIO(), io.BytesIO()
            torch.save(packed_vid, packed_stream)
            torch.save(legacy_vid, legacy_stream)
            quant_report["storage"]["packed_checkpoint_bytes"] = packed_stream.tell()
            quant_report["storage"]["packed_checkpoint_size_MB"] = packed_stream.tell() / (1024 ** 2)
            quant_report["storage"]["legacy_uint8_checkpoint_size_MB"] = legacy_stream.tell() / (1024 ** 2)
        args.packed_checkpoint_bytes = quant_report["storage"]["packed_checkpoint_bytes"]
        args.packed_checkpoint_size_MB = quant_report["storage"]["packed_checkpoint_size_MB"]
        args.legacy_uint8_checkpoint_size_MB = quant_report["storage"]["legacy_uint8_checkpoint_size_MB"]
        quant_vid = packed_vid if args.quant_storage_mode == "packed" else legacy_vid
        quant_path = f"{args.outf}/quant_vid.pth"
        atomic_torch_save(quant_vid, quant_path)
        base_model = unwrap_model(model)
        if isinstance(base_model, HNeRV):
            decoder_path = f"{args.outf}/img_decoder.pth"
            decoder_temporary = decoder_path + ".tmp"
            try:
                torch.jit.save(
                    torch.jit.trace(HNeRVDecoder(base_model), (vid_embed[:2])),
                    decoder_temporary)
                os.replace(decoder_temporary, decoder_path)
            finally:
                if os.path.exists(decoder_temporary):
                    os.remove(decoder_temporary)
        else:
            print("Skipping TorchScript decoder export for split ChromaHNeRV model.", flush=True)
        for metrics in metrics_by_variant.values():
            update_quant_fields(metrics, args)

    h, w = img_data.shape[-2:]
    return metrics_by_variant, (h, w)


def compute_quick_quality_metrics(pred_rgb, gt_rgb, pred_ycbcr, gt_ycbcr):
    eps = 1e-9
    mse_rgb = F.mse_loss(pred_rgb, gt_rgb, reduction="none").flatten(1).mean(1).detach().cpu()
    rgb_psnr = (-10 * torch.log10(mse_rgb + eps)).tolist()
    mse_y = F.mse_loss(
        pred_ycbcr[:, 0:1], gt_ycbcr[:, 0:1],
        reduction="none").flatten(1).mean(1).detach().cpu()
    y_psnr = (-10 * torch.log10(mse_y + eps)).tolist()
    return {
        "rgb_psnr": rgb_psnr,
        "psnr_y": y_psnr,
        "frame_psnr_mean": rgb_psnr,
        "frame_y_psnr_mean": y_psnr,
    }


def compute_quality_metrics(pred_rgb, gt_rgb, pred_ycbcr, gt_ycbcr):
    eps = 1e-9
    mse_rgb = F.mse_loss(pred_rgb, gt_rgb, reduction="none").flatten(1).mean(1).detach().cpu()
    rgb_psnr = (-10 * torch.log10(mse_rgb + eps)).tolist()
    rgb_ms_ssim = safe_msssim(pred_rgb, gt_rgb).tolist()

    mse_ch = F.mse_loss(pred_ycbcr, gt_ycbcr, reduction="none").mean(dim=(2, 3)).detach().cpu()
    psnr_ch = -10 * torch.log10(mse_ch + eps)
    psnr_y, psnr_cb, psnr_cr = [psnr_ch[:, i].tolist() for i in range(3)]
    yuv_psnr_dbavg = ((6 * psnr_ch[:, 0] + psnr_ch[:, 1] + psnr_ch[:, 2]) / 8).tolist()
    yuv_mse = (6 * mse_ch[:, 0] + mse_ch[:, 1] + mse_ch[:, 2]) / 8
    yuv_psnr_mse = (-10 * torch.log10(yuv_mse + eps)).tolist()

    ssim_y = safe_ssim(pred_ycbcr[:, 0:1], gt_ycbcr[:, 0:1]).tolist()
    ssim_cb = safe_ssim(pred_ycbcr[:, 1:2], gt_ycbcr[:, 1:2]).tolist()
    ssim_cr = safe_ssim(pred_ycbcr[:, 2:3], gt_ycbcr[:, 2:3]).tolist()
    yuv_ssim = [(6 * y + cb + cr) / 8 for y, cb, cr in zip(ssim_y, ssim_cb, ssim_cr)]
    return {
        "rgb_psnr": rgb_psnr, "rgb_ms_ssim": rgb_ms_ssim,
        "psnr_y": psnr_y, "psnr_cb": psnr_cb, "psnr_cr": psnr_cr,
        "yuv_psnr_611_dbavg": yuv_psnr_dbavg, "yuv_psnr_611_mse": yuv_psnr_mse,
        "ssim_y": ssim_y, "ssim_cb": ssim_cb, "ssim_cr": ssim_cr, "yuv_ssim_611": yuv_ssim,
        "frame_psnr_mean": rgb_psnr, "frame_y_psnr_mean": psnr_y,
    }


def summarize_metrics(args, metric_lists, frame_psnr, frame_y_psnr, temporal_diffs, lpips_vals):
    metrics = {name: safe_mean(metric_lists[name]) for name in metric_lists}
    metrics.update({
        "params_M": args.params_M,
        "encoder_params_M": args.encoder_params_M,
        "decoder_params_M": args.decoder_params_M,
        "embed_params_M": args.embed_params_M,
        "actual_total_params_M": args.actual_total_params_M,
        "modelsize_target_M": args.modelsize,
        "checkpoint_size_MB": checkpoint_size_mb(args),
        "estimated_gflops": getattr(args, "estimated_gflops", float("nan")),
        "quant_model_bit": args.quant_model_bit,
        "quant_embed_bit": args.quant_embed_bit,
        "bits_per_param": getattr(args, "bits_per_param", float("nan")),
        "bits_per_param_with_overhead": getattr(args, "full_bits_per_param", float("nan")),
        "bits_per_pixel": getattr(args, "total_bpp", float("nan")),
        "lpips_alex": safe_mean(lpips_vals) if lpips_vals else float("nan"),
        "frame_psnr_mean": safe_mean(frame_psnr),
        "frame_psnr_std": safe_std(frame_psnr),
        "frame_y_psnr_mean": safe_mean(frame_y_psnr),
        "frame_y_psnr_std": safe_std(frame_y_psnr),
        "temporal_rgb_error_diff": safe_mean(temporal_diffs),
    })
    metrics.update(architecture_metadata(args))
    return metrics


def build_profile_metrics(args):
    metrics = {name: float("nan") for name in METRIC_COLUMNS}
    metrics.update({
        "params_M": args.params_M,
        "encoder_params_M": args.encoder_params_M,
        "decoder_params_M": args.decoder_params_M,
        "embed_params_M": args.embed_params_M,
        "actual_total_params_M": args.actual_total_params_M,
        "modelsize_target_M": args.modelsize,
        "checkpoint_size_MB": float("nan"),
        "estimated_gflops": getattr(args, "estimated_gflops", float("nan")),
        "model_fps": float("nan"),
        "end_to_end_fps": float("nan"),
        "quant_model_bit": args.quant_model_bit,
        "quant_embed_bit": args.quant_embed_bit,
        "bits_per_param": float("nan"),
        "bits_per_param_with_overhead": float("nan"),
        "bits_per_pixel": float("nan"),
    })
    metrics.update(architecture_metadata(args))
    return metrics


def checkpoint_size_mb(args):
    for filename in ["model_latest.pth", f"epoch{getattr(args, 'cur_epoch', args.epochs)}.pth", "model_best.pth"]:
        path = os.path.join(args.outf, filename)
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 ** 2)
    return float("nan")


def update_quant_fields(metrics, args):
    metrics["bits_per_param"] = getattr(args, "bits_per_param", float("nan"))
    metrics["bits_per_param_with_overhead"] = getattr(args, "full_bits_per_param", float("nan"))
    metrics["bits_per_pixel"] = getattr(args, "total_bpp", float("nan"))
    for field in STORAGE_FIELDS:
        metrics[field] = getattr(args, field, float("nan"))


def huffman_payload_bits(records):
    by_group = {}
    for group, record in records:
        by_group.setdefault((group, int(record["bits"])), []).extend(record["quant"].flatten().tolist())
    payload, metadata = 0, 0
    for symbols in by_group.values():
        if not symbols:
            continue
        codec = HuffmanCodec.from_data(symbols)
        table = codec.get_code_table()
        lengths = {symbol: code[0] for symbol, code in table.items()}
        unique, counts = np.unique(symbols, return_counts=True)
        payload += sum(int(count) * lengths[int(symbol)] for symbol, count in zip(unique, counts))
        metadata += len(table) * 16 + 64  # Explicitly an estimated codebook/group header.
    return payload, metadata


def finalize_storage_report(args, report, quant_embed, quant_ckt, include_huffman):
    storage = report["storage"]
    storage["embedding_payload_bits"] = quant_embed["numel"] * quant_embed["bits"]
    embed_packed_bits = pack_nbit_tensor(quant_embed["quant"], quant_embed["bits"]).numel() * 8
    storage["tensor_padding_bits"] += embed_packed_bits - storage["embedding_payload_bits"]
    storage["scale_min_overhead_bits"] += (
        quant_embed["min"].numel() + quant_embed["scale"].numel()) * 16
    storage["metadata_bits"] += quant_tensor_metadata_bits(quant_embed)
    payload = sum(storage[f"{group}_payload_bits"] for group in ["shared", "y", "chroma", "rgb", "model"])
    packed_payload = payload + storage["embedding_payload_bits"] + storage["tensor_padding_bits"]
    total = packed_payload + storage["scale_min_overhead_bits"] + storage["buffer_bits"] + storage["metadata_bits"]
    storage["total_fixed_width_bits"] = total
    storage["quantized_param_count"] = sum(storage[f"{group}_param_count"] for group in ["shared", "y", "chroma", "rgb", "model"])
    stored_values = storage["quantized_param_count"] + quant_embed["numel"]
    storage["effective_bits_per_weight"] = total / max(storage["quantized_param_count"], 1)
    storage["effective_bits_per_stored_value"] = total / max(stored_values, 1)
    storage["fixed_width_bpp"] = total / args.final_size / args.full_data_length
    storage["fixed_width_weight_bits"] = payload
    storage["fixed_width_total_bits"] = total
    storage["effective_bits_per_playback_weight"] = total / max(storage["quantized_param_count"], 1)
    storage["embedding_bits"] = storage["embedding_payload_bits"]
    legacy = {
        "format_version": 2, "quant_config": quant_config(args), "embed": quant_embed,
        "model": quant_ckt, "buffers": report["buffers"], "storage_report": storage,
    }
    stream = io.BytesIO()
    torch.save(legacy, stream)
    storage["legacy_uint8_checkpoint_size_MB"] = stream.tell() / (1024 ** 2)
    if include_huffman:
        records = [(report["parameter_groups"][name], record) for name, record in quant_ckt.items()]
        records.append(("embedding", quant_embed))
        huff_payload, codec_estimate = huffman_payload_bits(records)
        huff_overhead = storage["scale_min_overhead_bits"] + storage["buffer_bits"] + storage["metadata_bits"] + codec_estimate
        storage["huffman_payload_bits"] = huff_payload
        storage["huffman_overhead_bits"] = huff_overhead
        storage["huffman_total_bits"] = huff_payload + huff_overhead
        storage["huffman_bpp"] = storage["huffman_total_bits"] / args.final_size / args.full_data_length
    args.bits_per_param = payload / max(storage["quantized_param_count"], 1)
    args.full_bits_per_param = storage["effective_bits_per_weight"]
    args.total_bpp = storage["fixed_width_bpp"]
    for key, value in storage.items():
        setattr(args, key, value)
    print_quant_report(report, args)


def dump_csv(args, metrics_by_variant, filename="results.csv"):
    row = build_result_row(args, metrics_by_variant)
    csv_path = os.path.join(args.outf, filename)
    print(f"results dumped to {csv_path}")
    write_csv_row(csv_path, row, append=False)


def append_consolidated_csv(args, metrics_by_variant):
    if not args.results_csv:
        return
    row = build_result_row(args, metrics_by_variant)
    write_csv_row(args.results_csv, row, append=True)


def build_result_row(args, metrics_by_variant):
    arch_meta = architecture_metadata(args)
    effective_bits = effective_quant_bits(args)
    row = {
        "run_name": args.run_name,
        "checkpoint_path": args.weight,
        "experiment": args.experiment,
        "architecture": arch_meta["architecture"],
        "chroma_format": arch_meta["chroma_format"],
        "split_stage": arch_meta["split_stage"],
        "branch_width": arch_meta["branch_width"],
        "branch_width_mode": arch_meta["branch_width_mode"],
        "chroma_scale": arch_meta["chroma_scale"],
        "chroma_downsample": args.chroma_downsample,
        "chroma_upsample": args.chroma_upsample,
        "vid": args.vid,
        "modelsize": args.modelsize,
        "epochs": args.epochs,
        "crop_list": args.crop_list,
        "resize_list": args.resize_list,
        "data_split": args.data_split,
        "dataset_preset": args.dataset_preset,
        "detected_frames": getattr(args, "detected_frames", float("nan")),
        "detected_source_height": getattr(args, "detected_source_height", float("nan")),
        "detected_source_width": getattr(args, "detected_source_width", float("nan")),
        "detected_final_height": getattr(args, "detected_final_height", float("nan")),
        "detected_final_width": getattr(args, "detected_final_width", float("nan")),
        "enc_strds": args.enc_strd_str,
        "dec_strds": args.dec_strd_str,
        "enc_dim": args.enc_dim,
        "ks": args.ks,
        "reduce": args.reduce,
        "lower_width": args.lower_width,
        "loss": args.loss,
        "lambda_y": args.lambda_y,
        "lambda_c": args.lambda_c,
        "lambda_rgb": args.lambda_rgb,
        "manualSeed": args.manualSeed,
        "cur_epoch": getattr(args, "cur_epoch", ""),
        "train_time": getattr(args, "train_time", ""),
        "quant_configuration": args.quant_str,
        "quant_scheme": args.quant_scheme,
        "quant_tag": args.quant_tag,
        "quant_matrix": args.quant_matrix,
        "rd_curve": args.rd_curve,
        "quant_shared_bit": effective_bits["shared"],
        "quant_y_bit": effective_bits["y"],
        "quant_chroma_bit": effective_bits["chroma"],
        "quant_rgb_bit": effective_bits["rgb"],
        "quant_model_bit": args.quant_model_bit,
        "quant_embed_bit": args.quant_embed_bit,
        "actual_total_params": getattr(args, "actual_total_params", float("nan")),
        "decoder_playback_params": getattr(args, "decoder_playback_params", float("nan")),
        "embedding_payload_values": int(round(getattr(args, "embed_params_M", 0) * 1e6)),
        "native_channel_schedule": getattr(args, "native_channel_schedule", ""),
        "shared_channels": getattr(args, "shared_channels", ""),
        "rgb_branch_channels": getattr(args, "rgb_branch_channels", ""),
        "y_branch_channels": getattr(args, "y_branch_channels", ""),
        "chroma_branch_channels": getattr(args, "chroma_branch_channels", ""),
        "rgb_or_y_output_resolution": args.crop_list,
        "chroma_output_resolution": (
            "_".join(str(int(value) // args.chroma_scale) for value in args.crop_list.split("_"))
            if is_chroma420_experiment(args) else args.crop_list),
        "decoder_stage_resolutions": args.decoder_stage_resolutions,
        "split_alias": args.split_alias,
        "split_stage_index": args.split_stage_index,
        "shared_output_height": args.shared_output_height,
        "shared_output_width": args.shared_output_width,
        "chroma_output_height": args.chroma_output_height,
        "chroma_output_width": args.chroma_output_width,
        "final_output_height": args.final_output_height,
        "final_output_width": args.final_output_width,
        "intermediate_eval_mode": args.intermediate_eval_mode,
        "final_eval_mode": args.final_eval_mode,
    }
    for group in ["shared", "rgb", "y", "chroma", "model"]:
        row[f"architecture_{group}_param_count"] = getattr(args, f"architecture_{group}_param_count", 0)
        row[f"architecture_{group}_param_percent"] = getattr(args, f"architecture_{group}_param_percent", 0.0)
    for field in STORAGE_FIELDS:
        row[field] = getattr(args, field, float("nan"))
    orig = metrics_by_variant.get("orig", {})
    for col in METRIC_COLUMNS:
        row[col] = orig.get(col, float("nan"))
    quant = metrics_by_variant.get("quant", {})
    for col in METRIC_COLUMNS:
        row[f"quant_{col}"] = quant.get(col, float("nan"))
    delta_map = {
        "rgb_psnr": "rgb_psnr", "y_psnr": "psnr_y", "cb_psnr": "psnr_cb",
        "cr_psnr": "psnr_cr", "yuv_psnr": "yuv_psnr_611_mse",
        "ms_ssim": "rgb_ms_ssim", "yuv_ssim": "yuv_ssim_611", "lpips": "lpips_alex",
    }
    for output_name, metric_name in delta_map.items():
        row[f"quant_{output_name}_delta"] = quant.get(metric_name, float("nan")) - orig.get(metric_name, float("nan"))
    return row


def write_csv_row(path, row, append):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)
    if append and file_exists:
        with open(path, newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            old_rows = list(reader)
            old_fields = reader.fieldnames or []
        new_fields = old_fields + [field for field in row if field not in old_fields]
        if new_fields != old_fields:
            with open(path, "w", newline="") as expanded_file:
                writer = csv.DictWriter(expanded_file, fieldnames=new_fields)
                writer.writeheader()
                writer.writerows(old_rows)
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        if append and file_exists:
            fieldnames = new_fields
        else:
            fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not append) or (not file_exists):
            writer.writeheader()
        writer.writerow(row)


def _strip_module_prefix(name):
    return name[7:] if name.startswith("module.") else name


def classify_quant_group(model, parameter_name, args):
    name = _strip_module_prefix(parameter_name)
    if name.startswith("encoder."):
        return "encoder"
    base_model = unwrap_model(model)
    if isinstance(base_model, ChromaHNeRV420):
        prefixes = {
            "shared_decoder.": "shared", "y_branch.": "y", "y_head.": "y",
            "cbcr_branch.": "chroma", "cbcr_head.": "chroma",
        }
    elif isinstance(base_model, RGBSplitHNeRV):
        prefixes = {"shared_decoder.": "shared", "rgb_branch.": "rgb", "rgb_head.": "rgb"}
    elif isinstance(base_model, HNeRV):
        prefixes = {"decoder.": "model", "head_layer.": "model"}
    else:
        raise TypeError(f"Unsupported quantized model type: {type(base_model).__name__}")
    matches = [group for prefix, group in prefixes.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Parameter '{parameter_name}' does not match exactly one quantization group.")
    return matches[0]


def resolve_group_bit(group, args):
    if args.quant_scheme == "uniform":
        return args.quant_model_bit
    mapping = {
        "shared": args.quant_shared_bit, "y": args.quant_y_bit,
        "chroma": args.quant_chroma_bit, "rgb": args.quant_rgb_bit,
    }
    if group not in mapping or mapping[group] not in SUPPORTED_QUANT_BITS:
        raise ValueError(f"No valid component bitwidth assigned to quantization group '{group}'.")
    return mapping[group]


def quant_config(args):
    effective = effective_quant_bits(args)
    return {
        "scheme": args.quant_scheme, "tag": args.quant_tag, "quant_axis": args.quant_axis,
        "model_bit": args.quant_model_bit, "embed_bit": args.quant_embed_bit,
        "shared_bit": effective["shared"], "y_bit": effective["y"],
        "chroma_bit": effective["chroma"], "rgb_bit": effective["rgb"],
        "storage_mode": args.quant_storage_mode,
    }


def effective_quant_bits(args):
    bits = {
        "shared": args.quant_shared_bit, "y": args.quant_y_bit,
        "chroma": args.quant_chroma_bit, "rgb": args.quant_rgb_bit,
    }
    if args.quant_scheme == "uniform":
        if args.experiment.startswith("rgbsplit_"):
            bits.update(shared=args.quant_model_bit, rgb=args.quant_model_bit)
        elif args.experiment.startswith("chroma420_"):
            bits.update(shared=args.quant_model_bit, y=args.quant_model_bit, chroma=args.quant_model_bit)
    return bits


def quant_tensor_metadata_bits(record):
    return 8 + 16 + 64 + 32 * len(record["shape"])


def empty_storage_report():
    report = {field: 0 for field in STORAGE_FIELDS}
    for field in ["huffman_payload_bits", "huffman_overhead_bits", "huffman_total_bits", "huffman_bpp",
                  "packed_checkpoint_size_MB", "legacy_uint8_checkpoint_size_MB"]:
        report[field] = float("nan")
    return report


def print_quant_report(report, args):
    storage = report["storage"]
    total_payload = sum(storage[f"{g}_payload_bits"] for g in ["shared", "y", "chroma", "rgb", "model"])
    print("Quantization group report:", flush=True)
    for group in ["shared", "y", "chroma", "rgb", "model"]:
        count = storage[f"{group}_param_count"]
        if not count:
            continue
        bits = report["group_bits"][group]
        pct = 100 * storage[f"{group}_payload_bits"] / max(total_payload, 1)
        tensors = report["group_tensor_counts"][group]
        print(f"  {group}: tensors={tensors}, parameters={count}, bits={bits}, "
              f"payload={storage[f'{group}_payload_bits']}, stored_weight_pct={pct:.2f}%", flush=True)
    print(f"  fixed_width_bpp={storage['fixed_width_bpp']:.6f}, "
          f"total_fixed_width_bits={storage['total_fixed_width_bits']}", flush=True)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1 and args.quant_scheme == "uniform":
        return model_list, None, None
    cur_model = deepcopy(model)
    cur_state = cur_model.state_dict()
    quant_ckt, parameter_groups = {}, {}
    group_counts = {group: 0 for group in ["shared", "y", "chroma", "rgb", "model"]}
    group_tensors = {group: 0 for group in group_counts}
    group_bits = {}

    named_parameters = dict(cur_model.named_parameters())
    non_encoder_count = 0
    for name, parameter in named_parameters.items():
        group = classify_quant_group(cur_model, name, args)
        if group == "encoder":
            continue
        bits = resolve_group_bit(group, args)
        record, dequantized = quant_tensor(parameter.detach(), bits, args.quant_axis)
        assert_finite_tensor(dequantized, name, "quant_model dequantized parameter")
        cur_state[name] = dequantized.to(parameter.dtype)
        quant_ckt[name] = record
        parameter_groups[name] = group
        group_counts[group] += parameter.numel()
        group_tensors[group] += 1
        group_bits[group] = bits
        non_encoder_count += parameter.numel()

    if sum(group_counts.values()) != non_encoder_count:
        raise RuntimeError("Quantization group parameter counts do not cover all non-encoder parameters.")

    buffers = {}
    for name, buffer in cur_model.named_buffers():
        if _strip_module_prefix(name).startswith("encoder."):
            continue
        if torch.is_floating_point(buffer):
            buffers[name] = buffer.detach().to(torch.float16).cpu()
            cur_state[name] = buffers[name].to(buffer.device, buffer.dtype)
        else:
            buffers[name] = buffer.detach().cpu()
    cur_model.load_state_dict(cur_state)
    model_list.append(cur_model)

    storage = empty_storage_report()
    for group, count in group_counts.items():
        storage[f"{group}_param_count"] = count
        storage[f"{group}_payload_bits"] = count * group_bits.get(group, 0)
    for record in quant_ckt.values():
        packed_bits = pack_nbit_tensor(record["quant"], record["bits"]).numel() * 8
        storage["tensor_padding_bits"] += packed_bits - record["numel"] * record["bits"]
        storage["scale_min_overhead_bits"] += (record["min"].numel() + record["scale"].numel()) * 16
        storage["metadata_bits"] += quant_tensor_metadata_bits(record)
    storage["buffer_bits"] = sum(buffer.numel() * (16 if torch.is_floating_point(buffer) else buffer.element_size() * 8)
                                 for buffer in buffers.values())
    storage["metadata_bits"] += sum(64 + 32 * buffer.dim() for buffer in buffers.values())
    report = {
        "storage": storage, "buffers": buffers, "parameter_groups": parameter_groups,
        "group_tensor_counts": group_tensors, "group_bits": group_bits,
    }
    return model_list, quant_ckt, report


if __name__ == "__main__":
    main()
