import argparse
import csv
import imageio
import math
import os
import random
import shutil
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
from model_all import HNeRV, HNeRVDecoder, TransformInput, VideoDataSet
from model_chroma_hnerv import ChromaHNeRV420A320, RGBSplitHNeRVA320


CHROMA_EXPERIMENTS = ("rgb444_hnerv", "ycbcr444_hnerv", "rgbsplit_a320", "chroma420_a320")
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


def build_parser():
    sanity_epilog = """
Sanity-check examples (small/debug only, not full experiments):
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment rgb444_hnerv --modelsize 0.35 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment ycbcr444_hnerv --modelsize 0.35 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment rgbsplit_a320 --modelsize 0.35 --branch_width 8 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
  python train_chroma_hnerv.py --data_path data/bunny --vid bunny --experiment chroma420_a320 --modelsize 0.35 --branch_width 8 --lambda_rgb 0.1 --enc_strds 5 4 4 2 2 --dec_strds 5 4 4 2 2 --ks 0_1_5 --crop_list 640_1280 --resize_list -1 --loss L2 -b 2 -e 1 --eval_freq 1 --debug
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
    parser.add_argument("--dump_images", action="store_true", default=False, help="dump prediction images")
    parser.add_argument("--dump_videos", action="store_true", default=False, help="concat predictions into video")
    parser.add_argument("--eval_fps", action="store_true", default=False, help="forward multiple times for fps")
    parser.add_argument("--encoder_file", default="", type=str, help="specify the embedding file")

    parser.add_argument("--manualSeed", type=int, default=1, help="manual seed")
    parser.add_argument("-d", "--distributed", action="store_true", default=False, help="distributed training")
    parser.add_argument("--debug", action="store_true", help="debug status, early train/eval")
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    parser.add_argument("--weight", default="None", type=str, help="pretrained weights")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the output dir")
    parser.add_argument("--outf", default="unify", help="folder to output images and model checkpoints")
    parser.add_argument("--suffix", default="", help="suffix str for outf")

    parser.add_argument("--experiment", default="rgb444_hnerv", choices=CHROMA_EXPERIMENTS)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--results_csv", default="", type=str)
    parser.add_argument("--lambda_y", default=1.0, type=float)
    parser.add_argument("--lambda_c", default=1.0, type=float)
    parser.add_argument("--lambda_rgb", default=0.0, type=float)
    parser.add_argument("--split_stage", default="a320", choices=["a320"])
    parser.add_argument("--branch_width", default=8, type=int)
    parser.add_argument("--chroma_scale", default=2, type=int)
    parser.add_argument("--chroma_downsample", default="area", choices=["area"])
    parser.add_argument("--chroma_upsample", default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument("--profile_only", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
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


def setup_output_dir(args):
    if args.debug:
        args.eval_freq = 1
        args.outf = "output/debug"
    else:
        args.outf = os.path.join("output", args.outf)
    args.enc_strd_str = ",".join([str(x) for x in args.enc_strds])
    args.dec_strd_str = ",".join([str(x) for x in args.dec_strds])
    extra_str = "Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}".format(
        args.modelsize, args.conv_type[0], args.enc_strd_str, args.conv_type[1],
        args.dec_strd_str, "" if args.norm == "none" else f"_{args.norm}",
        "_dist" if args.distributed else "", "_shuffle_data" if args.shuffle_data else "")
    args.quant_str = f"quant_M{args.quant_model_bit}_E{args.quant_embed_bit}"
    embed_str = f"{args.embed}_Dim{args.enc_dim}"
    run_prefix = f"{args.experiment}_" + (f"{args.run_name}_" if args.run_name else "")
    split_str = f"_split{args.split_stage}_w{args.branch_width}" if args.experiment in ["rgbsplit_a320", "chroma420_a320"] else ""
    args.exp_id = run_prefix + (
        f"{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}"
        f"_low{args.lower_width}_blk{args.num_blks}_e{args.epochs}_b{args.batchSize}_{args.quant_str}"
        f"_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{split_str}{args.suffix}"
    )
    args.outf = os.path.join(args.outf, args.exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print("Will overwrite the existing output dir!")
        shutil.rmtree(args.outf)
    os.makedirs(args.outf, exist_ok=True)


def data_to_gpu(x, device):
    return x.to(device)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def architecture_metadata(args):
    if args.experiment in ["rgb444_hnerv", "ycbcr444_hnerv"]:
        return {
            "architecture": "hnerv_444",
            "chroma_format": "444",
            "split_stage": "none",
            "branch_width": float("nan"),
            "chroma_scale": 1,
            "output_sample_ratio": 1.0,
        }
    if args.experiment == "rgbsplit_a320":
        return {
            "architecture": "rgbsplit_a320",
            "chroma_format": "rgb444",
            "split_stage": args.split_stage,
            "branch_width": args.branch_width,
            "chroma_scale": 1,
            "output_sample_ratio": 1.0,
        }
    if args.experiment == "chroma420_a320":
        return {
            "architecture": "chroma420_a320",
            "chroma_format": "420",
            "split_stage": args.split_stage,
            "branch_width": args.branch_width,
            "chroma_scale": args.chroma_scale,
            "output_sample_ratio": 0.5,
        }
    raise ValueError(f"Unsupported experiment: {args.experiment}")


def build_model(args):
    if args.experiment in ["rgb444_hnerv", "ycbcr444_hnerv"]:
        return HNeRV(args)
    if args.experiment == "rgbsplit_a320":
        return RGBSplitHNeRVA320(args)
    if args.experiment == "chroma420_a320":
        return ChromaHNeRV420A320(args)
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


def interpret_prediction(raw_output, args, aux=None):
    if args.experiment in ["rgb444_hnerv", "rgbsplit_a320"]:
        rgb = raw_output
        ycbcr = rgb_to_ycbcr_bt709(rgb)
    elif args.experiment == "ycbcr444_hnerv":
        ycbcr = raw_output
        rgb = ycbcr_to_rgb_bt709(ycbcr)
    elif args.experiment == "chroma420_a320":
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
    if args.experiment in ["rgb444_hnerv", "rgbsplit_a320"]:
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

    if args.experiment == "chroma420_a320":
        if "inpaint" in args.vid:
            raise NotImplementedError("Chroma420-HNeRV inpainting is not implemented in this first experiment.")
        if aux is None:
            raise ValueError("chroma420_a320 training requires auxiliary model outputs.")
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
        metrics_by_variant, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
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
            if args.experiment == "chroma420_a320":
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
            last_metrics, hw = evaluate(
                model, full_dataloader, local_rank, args,
                args.dump_vis if epoch == args.epochs - 1 else False,
                huffman_coding=(epoch == args.epochs - 1))
            if local_rank in [0, None]:
                cur_psnr = last_metrics["orig"].get("rgb_psnr", float("nan"))
                best_rgb_psnr = max(best_rgb_psnr, cur_psnr)
                writer.add_scalar(f"Val/rgb_psnr_{hw}", cur_psnr, epoch + 1)
                print_str = f"Eval at epoch {epoch + 1} for {hw}: rgb_psnr: {cur_psnr:.2f} best_rgb_psnr: {best_rgb_psnr:.2f}"
                print(print_str, flush=True)
                with open(f"{args.outf}/rank0.txt", "a") as f:
                    f.write(print_str + "\n")

        save_checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        if local_rank in [0, None]:
            torch.save(save_checkpoint, f"{args.outf}/model_latest.pth")
            if epoch + 1 == args.epochs:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                if last_metrics is None:
                    last_metrics, _ = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
                dump_csv(args, last_metrics, f"epoch{epoch + 1}.csv")
                append_consolidated_csv(args, last_metrics)
                torch.save(save_checkpoint, f"{args.outf}/epoch{epoch + 1}.pth")
                if last_metrics["orig"].get("rgb_psnr", -float("inf")) >= best_rgb_psnr:
                    torch.save(save_checkpoint, f"{args.outf}/model_best.pth")

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


def load_checkpoint_if_needed(model, optimizer, args, local_rank):
    checkpoint = None
    if args.weight != "None":
        print(f"=> loading checkpoint '{args.weight}'")
        checkpoint = torch.load(args.weight, map_location="cpu")
        orig_ckt = checkpoint["state_dict"]
        new_ckt = {k.replace("blocks.0.", ""): v for k, v in orig_ckt.items()}
        if "module" in list(orig_ckt.keys())[0] and not hasattr(model, "module"):
            new_ckt = {k.replace("module.", ""): v for k, v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif "module" not in list(orig_ckt.keys())[0] and hasattr(model, "module"):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
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
def evaluate(model, full_dataloader, local_rank, args, dump_vis=False, huffman_coding=False):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metrics_by_variant = {}
    dequant_vid_embed = None
    quant_embed = None
    lpips_model = build_lpips_model(next(model.parameters()).device)

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
            if args.experiment == "chroma420_a320":
                raw_output, embed_list, dec_time, aux = cur_model(cur_input, input_embed, return_aux=True)
            else:
                raw_output, embed_list, dec_time = cur_model(cur_input, input_embed)
                aux = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_time += time.time() - fwd_start

            if model_ind == 0:
                img_embed_list.append(embed_list[0])
            if args.eval_fps:
                for _ in range(100):
                    if args.experiment == "chroma420_a320":
                        raw_output, embed_list, _, aux = cur_model(cur_input, embed_list[0], return_aux=True)
                    else:
                        raw_output, embed_list, _ = cur_model(cur_input, embed_list[0])

            pred = interpret_prediction(raw_output, args, aux)
            pred_rgb = pred["rgb"].clamp(0, 1)
            pred_ycbcr = pred["ycbcr"].clamp(0, 1)
            gt_rgb = img_gt.clamp(0, 1)
            gt_ycbcr = rgb_to_ycbcr_bt709(gt_rgb).clamp(0, 1)
            end_time += time.time() - step_start
            frame_count += gt_rgb.size(0)

            batch_metrics = compute_quality_metrics(pred_rgb, gt_rgb, pred_ycbcr, gt_ycbcr)
            for key, values in batch_metrics.items():
                metric_lists[key].extend(values)
            frame_psnr.extend(batch_metrics["frame_psnr_mean"])
            frame_y_psnr.extend(batch_metrics["frame_y_psnr_mean"])

            if lpips_model is not None:
                try:
                    lp = lpips_model(pred_rgb * 2 - 1, gt_rgb * 2 - 1).flatten().detach().cpu().tolist()
                    lpips_vals.extend(lp)
                except Exception:
                    pass

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
                fps = frame_count / max(fwd_time, 1e-12)
                print_str = (
                    f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Rank:{local_rank}, "
                    f"Eval {variant} Step [{i + 1}/{len(full_dataloader)}], FPS {round(fps, 1)}, "
                    f"rgb_psnr: {safe_mean(metric_lists['rgb_psnr']):.2f}"
                )
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open(f"{args.outf}/rank0.txt", "a") as f:
                        f.write(print_str + "\n")

        metrics = summarize_metrics(args, metric_lists, frame_psnr, frame_y_psnr, temporal_diffs, lpips_vals)
        metrics["model_fps"] = frame_count / max(fwd_time, 1e-12) if frame_count else float("nan")
        metrics["end_to_end_fps"] = frame_count / max(end_time, 1e-12) if frame_count else float("nan")
        metrics_by_variant[variant] = metrics

        if model_ind == 0 and args.quant_model_bit != -1:
            vid_embed = torch.cat(img_embed_list, 0)
            quant_embed, dequant_embed = quant_tensor(vid_embed, args.quant_embed_bit)
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
        quant_vid = {"embed": quant_embed, "model": quant_ckt}
        torch.save(quant_vid, f"{args.outf}/quant_vid.pth")
        base_model = unwrap_model(model)
        if isinstance(base_model, HNeRV):
            torch.jit.save(torch.jit.trace(HNeRVDecoder(base_model), (vid_embed[:2])), f"{args.outf}/img_decoder.pth")
        else:
            print("Skipping TorchScript decoder export for split ChromaHNeRV model.", flush=True)
        if huffman_coding:
            attach_huffman_bpp(args, quant_embed, quant_ckt)
            if "orig" in metrics_by_variant:
                update_quant_fields(metrics_by_variant["orig"], args)
            if "quant" in metrics_by_variant:
                update_quant_fields(metrics_by_variant["quant"], args)

    h, w = img_data.shape[-2:]
    return metrics_by_variant, (h, w)


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


def attach_huffman_bpp(args, quant_embed, quant_ckt):
    quant_v_list = quant_embed["quant"].flatten().tolist()
    tmin_scale_len = quant_embed["min"].nelement() + quant_embed["scale"].nelement()
    for _, layer_wt in quant_ckt.items():
        quant_v_list.extend(layer_wt["quant"].flatten().tolist())
        tmin_scale_len += layer_wt["min"].nelement() + layer_wt["scale"].nelement()

    unique, counts = np.unique(quant_v_list, return_counts=True)
    num_freq = dict(zip(unique, counts))
    codec = HuffmanCodec.from_data(quant_v_list)
    sym_bit_dict = {k: v[0] for k, v in codec.get_code_table().items()}
    total_bits = sum(freq * sym_bit_dict[num] for num, freq in num_freq.items())
    args.bits_per_param = total_bits / len(quant_v_list)
    total_bits += tmin_scale_len * 16
    args.full_bits_per_param = total_bits / len(quant_v_list)
    args.total_bpp = total_bits / args.final_size / args.full_data_length
    print(
        "After quantization and encoding: \n"
        f" bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}",
        flush=True)


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
    row = {
        "run_name": args.run_name,
        "experiment": args.experiment,
        "architecture": arch_meta["architecture"],
        "chroma_format": arch_meta["chroma_format"],
        "split_stage": arch_meta["split_stage"],
        "branch_width": arch_meta["branch_width"],
        "chroma_scale": arch_meta["chroma_scale"],
        "chroma_downsample": args.chroma_downsample,
        "chroma_upsample": args.chroma_upsample,
        "vid": args.vid,
        "modelsize": args.modelsize,
        "epochs": args.epochs,
        "crop_list": args.crop_list,
        "resize_list": args.resize_list,
        "data_split": args.data_split,
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
    }
    orig = metrics_by_variant.get("orig", {})
    for col in METRIC_COLUMNS:
        row[col] = orig.get(col, float("nan"))
    quant = metrics_by_variant.get("quant", {})
    for col in METRIC_COLUMNS:
        row[f"quant_{col}"] = quant.get(col, float("nan"))
    return row


def write_csv_row(path, row, append):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if (not append) or (not file_exists):
            writer.writeheader()
        writer.writerow(row)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    cur_model = deepcopy(model)
    quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
    encoder_k_list = []
    for k, v in cur_ckt.items():
        if "encoder" in k:
            encoder_k_list.append(k)
        else:
            quant_v, new_v = quant_tensor(v, args.quant_model_bit)
            quant_ckt[k] = quant_v
            cur_ckt[k] = new_v
    for encoder_k in encoder_k_list:
        del quant_ckt[encoder_k]
    cur_model.load_state_dict(cur_ckt)
    model_list.append(cur_model)
    return model_list, quant_ckt


if __name__ == "__main__":
    main()
