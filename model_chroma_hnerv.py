import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnerv_utils import ycbcr_to_rgb_bt709
from model_all import HNeRV, NeRVBlock, OutImg


def _branch_kernel(args, stage_idx):
    _, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split("_")]
    return min(ks_dec1 + 2 * stage_idx, ks_dec2)


def _decoder_channels(args):
    channels = []
    ngf = args.fc_dim
    for strd in args.dec_strds:
        reduction = strd ** 0.5 if args.reduce == -1 else args.reduce
        ngf = int(max(round(ngf / reduction), args.lower_width))
        channels.append(ngf)
    return channels


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


def _make_branch(args, in_channels, out_channels, stage_indices):
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
        self.split_stage = args.split_stage
        self.rgb_or_y_stage_indices = split["rgb_or_y_stage_indices"]
        self.cbcr_stage_indices = split["cbcr_stage_indices"]

        _, dec_blks = [int(x) for x in args.num_blks.split("_")]
        split_block_count = 1 + split["shared_count"] * dec_blks
        self.shared_decoder = nn.ModuleList(list(base.decoder[:split_block_count]))
        channel_schedule = _decoder_channels(args)
        self.shared_channels = channel_schedule[split["shared_count"] - 1]

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
        self.rgb_branch = _make_branch(
            args, self.shared_channels, args.branch_width, self.rgb_or_y_stage_indices)
        self.rgb_head = nn.Conv2d(args.branch_width, 3, 3, 1, 1)

    def forward(self, input, input_embed=None, encode_only=False):
        dec_start = time.time()
        shared, embed_list = self._shared_forward(input, input_embed)
        rgb_feat = self.rgb_branch(shared)
        embed_list.append(rgb_feat)
        img_out = OutImg(self.rgb_head(rgb_feat), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time


class ChromaHNeRV420(_SplitBase):
    def __init__(self, args):
        super().__init__(args)
        if args.chroma_scale != 2:
            raise NotImplementedError("Chroma420 split models only support --chroma_scale 2.")
        self.chroma_upsample = args.chroma_upsample
        self.y_branch = _make_branch(
            args, self.shared_channels, args.branch_width, self.rgb_or_y_stage_indices)
        self.y_head = nn.Conv2d(args.branch_width, 1, 3, 1, 1)
        self.cbcr_branch = _make_branch(
            args, self.shared_channels, args.branch_width, self.cbcr_stage_indices)
        cbcr_head_channels = args.branch_width if self.cbcr_stage_indices else self.shared_channels
        self.cbcr_head = nn.Conv2d(cbcr_head_channels, 2, 3, 1, 1)

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

        if torch.cuda.is_available():
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
