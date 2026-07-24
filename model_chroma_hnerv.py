import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnerv_utils import ycbcr_to_rgb_bt709
from model_all import HNeRV, NeRVBlock, OutImg, decoder_channel_schedule


def _branch_kernel(args, stage_idx):
    _, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split("_")]
    return min(ks_dec1 + 2 * stage_idx, ks_dec2)


def _split_config(args):
    stage_count = len(args.dec_strds)
    if args.split_stage == "a320":
        shared_count = stage_count - 1
        return {
            "shared_count": shared_count,
            "rgb_or_y_stage_indices": [stage_count - 1],
            "cbcr_stage_indices": [],
        }
    if args.split_stage == "a160":
        shared_count = stage_count - 2
        return {
            "shared_count": shared_count,
            "rgb_or_y_stage_indices": [stage_count - 2, stage_count - 1],
            "cbcr_stage_indices": [stage_count - 2],
        }
    raise NotImplementedError(f"Unsupported split stage: {args.split_stage}")


def _make_fixed_branch(args, in_channels, out_channels, stage_indices):
    _, dec_blks = [int(x) for x in args.num_blks.split("_")]
    blocks = []
    cur_in = in_channels
    for stage_idx in stage_indices:
        stage_stride = args.dec_strds[stage_idx]
        stage_kernel = _branch_kernel(args, stage_idx)
        for block_idx in range(dec_blks):
            blocks.append(NeRVBlock(
                dec_block=True,
                conv_type=args.conv_type[1],
                ngf=cur_in,
                new_ngf=out_channels,
                ks=stage_kernel,
                strd=stage_stride if block_idx == 0 else 1,
                bias=True,
                norm=args.norm,
                act=args.act,
            ))
            cur_in = out_channels
    return nn.Sequential(*blocks) if blocks else nn.Identity()


def _stage_module_slice(decoder, args, stage_indices):
    if not stage_indices:
        return []
    _, dec_blks = [int(x) for x in args.num_blks.split("_")]
    modules = []
    for stage_idx in stage_indices:
        start = 1 + stage_idx * dec_blks
        modules.extend(decoder[start:start + dec_blks])
    return modules


def convert_full_state_dict_to_native_rgbsplit(state_dict, dec_strds, num_blks, split_stage):
    """Convert Full HNeRV state keys into the structurally equivalent native RGBSplit layout."""
    stage_count = len(dec_strds)
    shared_count = stage_count - (1 if split_stage == "a320" else 2 if split_stage == "a160" else 0)
    if shared_count <= 0 or shared_count >= stage_count:
        raise ValueError(f"Unsupported or invalid split stage: {split_stage}")
    _, dec_blks = [int(x) for x in num_blks.split("_")]
    split_block_count = 1 + shared_count * dec_blks
    converted = {}
    for raw_name, value in state_dict.items():
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        if name.startswith("blocks.0."):
            name = name[len("blocks.0."):]
        if name.startswith("decoder."):
            parts = name.split(".", 2)
            block_index = int(parts[1])
            suffix = parts[2] if len(parts) == 3 else ""
            if block_index < split_block_count:
                new_name = f"shared_decoder.{block_index}"
            else:
                new_name = f"rgb_branch.{block_index - split_block_count}"
            converted[f"{new_name}.{suffix}" if suffix else new_name] = value
        elif name.startswith("head_layer."):
            converted[name.replace("head_layer.", "rgb_head.", 1)] = value
        else:
            converted[name] = value
    if not any(name.startswith("shared_decoder.") for name in converted):
        raise ValueError("Full HNeRV state dict contains no decoder parameters.")
    if not any(name.startswith("rgb_branch.") for name in converted):
        raise ValueError("Split conversion produced no RGB branch parameters.")
    if "rgb_head.weight" not in converted:
        raise ValueError("Full HNeRV state dict is missing head_layer.weight.")
    return converted


class _SplitBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        split = _split_config(args)
        if split["shared_count"] < 1:
            raise ValueError(f"{args.split_stage} split leaves no shared upsample stages.")
        if split["shared_count"] >= len(args.dec_strds):
            raise ValueError(f"{args.split_stage} split leaves no branch stages.")

        base = HNeRV(args)
        self.embed = base.embed
        self.encoder = base.encoder
        if hasattr(base, "pe_embed"):
            self.pe_embed = base.pe_embed
        self.fc_h, self.fc_w = base.fc_h, base.fc_w
        self.out_bias = base.out_bias
        self.measure_latency = getattr(args, "measure_latency", False) or getattr(args, "eval_fps", False)
        self.split_stage = args.split_stage
        self.branch_width_mode = getattr(args, "branch_width_mode", "fixed")
        self.rgb_or_y_stage_indices = split["rgb_or_y_stage_indices"]
        self.cbcr_stage_indices = split["cbcr_stage_indices"]

        _, dec_blks = [int(x) for x in args.num_blks.split("_")]
        split_block_count = 1 + split["shared_count"] * dec_blks
        self.shared_decoder = nn.ModuleList(list(base.decoder[:split_block_count]))
        channel_schedule = decoder_channel_schedule(args)
        self.native_channel_schedule = tuple(channel_schedule)
        self.shared_channels = channel_schedule[split["shared_count"] - 1]
        self.shared_channel_schedule = tuple(channel_schedule[:split["shared_count"]])
        object.__setattr__(self, "_base_decoder_ref", base.decoder)

    def _make_branch(self, args, stage_indices):
        if self.branch_width_mode == "native":
            modules = _stage_module_slice(self._base_decoder_ref, args, stage_indices)
            return nn.Sequential(*modules) if modules else nn.Identity()
        return _make_fixed_branch(args, self.shared_channels, args.branch_width, stage_indices)

    def _branch_schedule(self, args, stage_indices):
        if self.branch_width_mode == "native":
            return tuple(self.native_channel_schedule[index] for index in stage_indices)
        return tuple(args.branch_width for _ in stage_indices)

    def _encode(self, input, input_embed=None):
        if input_embed is not None:
            return input_embed
        if "pe" in self.embed:
            input = self.pe_embed(input[:, None]).float()
        return self.encoder(input)

    def _shared_forward(self, input, input_embed=None):
        img_embed = self._encode(input, input_embed)
        embed_list = [img_embed]
        output = self.shared_decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(
            0, 1, 4, 2, 5, 3).reshape(n, -1, self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.shared_decoder[1:]:
            output = layer(output)
            embed_list.append(output)
        return output, embed_list


class RGBSplitHNeRV(_SplitBase):
    def __init__(self, args):
        super().__init__(args)
        self.rgb_branch = self._make_branch(args, self.rgb_or_y_stage_indices)
        self.rgb_branch_channels = self._branch_schedule(args, self.rgb_or_y_stage_indices)
        head_channels = self.rgb_branch_channels[-1]
        self.rgb_head = nn.Conv2d(head_channels, 3, 3, 1, 1)
        object.__delattr__(self, "_base_decoder_ref")

    def forward(self, input, input_embed=None, encode_only=False):
        dec_start = time.time()
        shared, embed_list = self._shared_forward(input, input_embed)
        rgb_feat = self.rgb_branch(shared)
        embed_list.append(rgb_feat)
        img_out = OutImg(self.rgb_head(rgb_feat), self.out_bias)
        if self.measure_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time


class ChromaHNeRV420(_SplitBase):
    def __init__(self, args):
        super().__init__(args)
        if args.chroma_scale != 2:
            raise NotImplementedError("Chroma420 split models only support --chroma_scale 2.")
        self.chroma_upsample = args.chroma_upsample
        self.y_branch = self._make_branch(args, self.rgb_or_y_stage_indices)
        self.y_branch_channels = self._branch_schedule(args, self.rgb_or_y_stage_indices)
        self.y_head = nn.Conv2d(self.y_branch_channels[-1], 1, 3, 1, 1)
        if self.branch_width_mode == "native":
            cbcr_modules = deepcopy(_stage_module_slice(self._base_decoder_ref, args, self.cbcr_stage_indices))
            self.cbcr_branch = nn.Sequential(*cbcr_modules) if cbcr_modules else nn.Identity()
        else:
            self.cbcr_branch = self._make_branch(args, self.cbcr_stage_indices)
        self.chroma_branch_channels = self._branch_schedule(args, self.cbcr_stage_indices)
        cbcr_head_channels = self.chroma_branch_channels[-1] if self.chroma_branch_channels else self.shared_channels
        self.cbcr_head = nn.Conv2d(cbcr_head_channels, 2, 3, 1, 1)
        object.__delattr__(self, "_base_decoder_ref")

    def _upsample_cbcr(self, cbcr_low, target_hw):
        if self.chroma_upsample == "nearest":
            return F.interpolate(cbcr_low, size=target_hw, mode="nearest")
        if self.chroma_upsample == "bilinear":
            return F.interpolate(cbcr_low, size=target_hw, mode="bilinear", align_corners=False)
        raise NotImplementedError(f"Unsupported chroma upsample mode: {self.chroma_upsample}")

    def forward(self, input, input_embed=None, encode_only=False, return_aux=False):
        dec_start = time.time()
        shared, embed_list = self._shared_forward(input, input_embed)

        cbcr_feat = self.cbcr_branch(shared)
        cbcr_low = OutImg(self.cbcr_head(cbcr_feat), self.out_bias)
        y_feat = self.y_branch(shared)
        embed_list.extend([cbcr_feat, y_feat])
        y = OutImg(self.y_head(y_feat), self.out_bias)
        cbcr_up = self._upsample_cbcr(cbcr_low, y.shape[-2:])
        ycbcr = torch.cat([y, cbcr_up], dim=1)
        rgb = ycbcr_to_rgb_bt709(ycbcr).clamp(0, 1)

        if self.measure_latency and torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        if return_aux:
            aux = {
                "rgb": rgb,
                "ycbcr": ycbcr,
                "y": y,
                "cbcr_low": cbcr_low,
                "cbcr_up": cbcr_up,
            }
            return rgb, embed_list, dec_time, aux
        return rgb, embed_list, dec_time


class RGBSplitHNeRVA320(RGBSplitHNeRV):
    pass


class ChromaHNeRV420A320(ChromaHNeRV420):
    pass


class RGBSplitHNeRVA160(RGBSplitHNeRV):
    pass


class ChromaHNeRV420A160(ChromaHNeRV420):
    pass
