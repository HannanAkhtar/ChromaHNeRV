import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

################## split one video into seen/unseen frames ##################
def data_split(img_list, split_num_list, shuffle_data, rand_num=0):
    valid_train_length, total_train_length, total_data_length = split_num_list
    # assert total_train_length < total_data_length
    temp_train_list, temp_val_list = [], []
    if shuffle_data:
        random.Random(rand_num).shuffle(img_list)
    for cur_i, frame_id in enumerate(img_list):
        if (cur_i % total_data_length) < valid_train_length:
            temp_train_list.append(frame_id)
        elif (cur_i % total_data_length) >= total_train_length:
            temp_val_list.append(frame_id)
    return temp_train_list, temp_val_list

################# Tensor quantization and dequantization #################
def _safe_quant_scale(scale, eps=1e-12):
    return torch.where(
        torch.isfinite(scale) & (scale.abs() > eps),
        scale,
        torch.ones_like(scale),
    )


def _quant_min_scale(t, bits, quant_axis):
    if quant_axis == -1 or t.dim() == 0:
        t_min, t_max = t.min(), t.max()
    else:
        if not 0 <= quant_axis < t.dim():
            raise ValueError(f"quant_axis must be -2, -1, or a tensor axis, got {quant_axis}.")
        reduce_dims = tuple(dim for dim in range(t.dim()) if dim != quant_axis)
        if not reduce_dims:
            t_min, t_max = t, t
        else:
            t_min = t.amin(dim=reduce_dims, keepdim=True)
            t_max = t.amax(dim=reduce_dims, keepdim=True)
    return t_min, _safe_quant_scale((t_max - t_min) / (2**bits - 1))


def quant_tensor(t, bits=8, quant_axis=0):
    if bits not in range(1, 9):
        raise ValueError(f"Quantization supports 1 through 8 bits, got {bits}.")
    if not torch.is_floating_point(t):
        raise TypeError(f"quant_tensor requires a floating tensor, got {t.dtype}.")
    if not torch.isfinite(t).all():
        raise ValueError("Cannot quantize a tensor containing NaN or Inf.")

    candidate_axes = [-1]
    if quant_axis == -2:
        candidate_axes.extend(range(t.dim()))
    else:
        candidate_axes = [quant_axis]

    candidates = []
    for axis in candidate_axes:
        t_min, scale = _quant_min_scale(t, bits, axis)
        quant = ((t - t_min) / scale).round().clamp(0, 2**bits - 1).to(torch.uint8)
        dequant = t_min + scale * quant.to(t.dtype)
        dequant = torch.nan_to_num(dequant, nan=0.0, posinf=0.0, neginf=0.0)
        candidates.append(((t - dequant).abs().mean().item(), axis, t_min, scale, quant, dequant))

    _, axis, t_min, scale, quant, dequant = min(candidates, key=lambda item: item[0])
    record = {
        "quant": quant,
        "min": t_min.to(torch.float16),
        "scale": scale.to(torch.float16),
        "bits": int(bits),
        "shape": tuple(t.shape),
        "numel": t.numel(),
        "quant_axis": int(axis),
    }
    # Evaluate the exact FP16 min/scale representation that will be serialized.
    return record, dequant_tensor(record).to(t.dtype)


def pack_nbit_tensor(values, bits):
    if bits not in range(1, 9):
        raise ValueError(f"Packing supports 1 through 8 bits, got {bits}.")
    values = values.detach().to(device="cpu", dtype=torch.int64).flatten()
    if values.numel() and ((values < 0).any() or (values > 2**bits - 1).any()):
        raise ValueError(f"Symbols must lie in [0, {2**bits - 1}] for {bits}-bit packing.")
    if not values.numel():
        return torch.empty(0, dtype=torch.uint8)
    shifts = torch.arange(bits, dtype=torch.int64)
    bitstream = ((values[:, None] >> shifts) & 1).flatten()
    padding = (-bitstream.numel()) % 8
    if padding:
        bitstream = F.pad(bitstream, (0, padding))
    byte_weights = (1 << torch.arange(8, dtype=torch.int64))
    return (bitstream.reshape(-1, 8) * byte_weights).sum(1).to(torch.uint8)


def unpack_nbit_tensor(packed, bits, numel, shape):
    if bits not in range(1, 9):
        raise ValueError(f"Unpacking supports 1 through 8 bits, got {bits}.")
    expected_bytes = math.ceil(int(numel) * bits / 8)
    packed = packed.detach().to(device="cpu", dtype=torch.uint8).flatten()
    if packed.numel() != expected_bytes:
        raise ValueError(f"Packed tensor has {packed.numel()} bytes; expected {expected_bytes}.")
    if int(numel) == 0:
        return torch.empty(tuple(shape), dtype=torch.uint8)
    shifts = torch.arange(8, dtype=torch.int64)
    bitstream = ((packed.to(torch.int64)[:, None] >> shifts) & 1).flatten()[:int(numel) * bits]
    symbol_weights = (1 << torch.arange(bits, dtype=torch.int64))
    values = (bitstream.reshape(int(numel), bits) * symbol_weights).sum(1)
    return values.to(torch.uint8).reshape(tuple(shape))


def serialize_quant_tensor(record):
    required = {"quant", "min", "scale", "bits", "shape", "numel", "quant_axis"}
    missing = required.difference(record)
    if missing:
        raise KeyError(f"Quantized tensor record is missing: {sorted(missing)}")
    return {
        "packed": pack_nbit_tensor(record["quant"], int(record["bits"])),
        "bits": int(record["bits"]),
        "shape": tuple(record["shape"]),
        "numel": int(record["numel"]),
        "min": record["min"].to(torch.float16).cpu(),
        "scale": record["scale"].to(torch.float16).cpu(),
        "quant_axis": int(record["quant_axis"]),
    }


def dequant_tensor(record):
    if "quant" in record:
        quant = record["quant"]
    elif "packed" in record:
        quant = unpack_nbit_tensor(
            record["packed"], int(record["bits"]), int(record["numel"]), record["shape"])
        quant = quant.to(record["min"].device)
    else:
        raise KeyError("Quantized tensor record must contain either 'quant' or 'packed'.")
    tmin, scale = record["min"].to(torch.float32), record["scale"].to(torch.float32)
    scale = _safe_quant_scale(scale.expand_as(quant))
    new_t = tmin.expand_as(quant) + scale * quant.to(torch.float32)
    new_t = torch.nan_to_num(new_t, nan=0.0, posinf=0.0, neginf=0.0)
    return new_t

################# Function used in distributed training #################
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


def adjust_lr(optimizer, cur_epoch, args):
    # cur_epoch = (cur_epoch + cur_iter) / args.epochs
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult


############################ Function for loss compuation and evaluate metrics ############################

def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr


def loss_fn(pred, target, loss_type='L2', batch_average=True):
    target = target.detach()

    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=False)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion2':
        loss = 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion4':
        loss = 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion6':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion9':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion10':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion11':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion12':
        loss = 0.8 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    return loss.mean() if batch_average else loss


def psnr_fn_single(output, gt):
    l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()

def psnr_fn_batch(output_list, gt):
    psnr_list = [psnr_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(psnr_list, 0).cpu()


def msssim_fn_single(output, gt):
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()

def msssim_fn_batch(output_list, gt):
    msssim_list = [msssim_fn_single(output.detach(), gt.detach()) for output in output_list]
    # for output in output_list:
    #     msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    #     msssim_list.append(msssim)
    return torch.stack(msssim_list, 0).cpu()


def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss + 1e-9)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr


def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim



############################ LEGACY CODE ############################
class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)
            # sin_Value = torch.sin(pos * self.lbase ** torch.arange(self.levels) * math.pi)
            # cos_Value = torch.cos(pos * self.lbase ** torch.arange(self.levels) * math.pi)
            # return torch.cat([sin_Value, cos_Value], dim=-1)


class PositionalEncodingTrans(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        index = torch.round(pos * self.max_len).long()
        p = self.pe[index]
        return p
def Diff_decompose(img_diff, residual_list=[], max_v=256, residual_depth=1):
    img_diff = (img_diff * 255.).round()
    decomse_diff = torch.zeros_like(img_diff)
    max_bit = int(np.log2(max_v))
    for i in range(max_bit):
        min_diff, max_diff = 2**i, 2**(i+1)
        decomse_diff[(min_diff<img_diff) & (img_diff<=max_diff)] = (min_diff + max_diff) / 2
        decomse_diff[(-min_diff>img_diff) & (img_diff>=-max_diff)] = -(min_diff + max_diff) / 2
    cur_max_v = max_v // 4
    residual_list.append(decomse_diff / 255.)
    if cur_max_v < 4 or len(residual_list)==residual_depth:
        return residual_list
    else:
        return Diff_decompose(img_diff - decomse_diff, residual_list, cur_max_v)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


def rgb_to_ycbcr_bt709(rgb: torch.Tensor) -> torch.Tensor:
    """Convert full-range normalized RGB tensors to BT.709 YCbCr."""
    if not isinstance(rgb, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rgb)}")
    if len(rgb.shape) < 3 or rgb.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {rgb.shape}")

    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556 + 0.5
    cr = (r - y) / 1.5748 + 0.5
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb_bt709(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert full-range normalized BT.709 YCbCr tensors to RGB."""
    if not isinstance(ycbcr, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(ycbcr)}")
    if len(ycbcr.shape) < 3 or ycbcr.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {ycbcr.shape}")

    y = ycbcr[..., 0, :, :]
    cb = ycbcr[..., 1, :, :]
    cr = ycbcr[..., 2, :, :]
    r = y + 1.5748 * (cr - 0.5)
    b = y + 1.8556 * (cb - 0.5)
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return torch.stack([r, g, b], -3).clamp(0, 1)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)



def eval_quantize_per_tensor(t, bit=8):
    tmin_scale_list = []
    # quantize on the full tensor
    tmin, t_max = t.min().expand_as(t), t.max().expand_as(t)
    scale = (t_max - t_min) / 2**bit
    tmin_scale_list.append([t_min, scale])

    # quantize on axis 0
    min_max_list = []
    for i in range(t.size(0)):
        t_valid = t[i]!=0
        if t_valid.sum():
            min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
        else:
            min_max_list.append([0, 0])
    min_max_tf = torch.tensor(min_max_list).to(t.device)        
    scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
    if t.dim() == 4:
        scale = scale[:,None,None,None]
        t_min = min_max_tf[:,0,None,None,None]
    elif t.dim() == 2:
        scale = scale[:,None]
        t_min = min_max_tf[:,0,None]
    tmin_scale_list.append([t_min, scale])

    # quantize on axis 1
    min_max_list = []
    for i in range(t.size(1)):
        t_valid = t[:,i]!=0
        if t_valid.sum():
            min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
        else:
            min_max_list.append([0, 0])
    min_max_tf = torch.tensor(min_max_list).to(t.device)             
    scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
    if t.dim() == 4:
        scale = scale[None,:,None,None]
        t_min = min_max_tf[None,:,0,None,None]
    elif t.dim() == 2:
        scale = scale[None,:]
        t_min = min_max_tf[None,:,0]    
    tmin_scale_list.append([t_min, scale])

    # import pdb; pdb.set_trace; from IPython import embed; embed()  
    quant_t_list, new_t_list, err_t_list = [], [], []
    for tmin, scale in tmin_scale_list:
        quant_t = ((t - tmin) / (scale + 1e-19)).round()
        new_t = tmin + scale * quant_t
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append((t - new_t).abs().mean())   

    # choose the best quantization way
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_quant_t = quant_t_list[best_quant_idx]
    best_new_t = new_t_list[best_quant_idx]

    return best_quant_t, best_new_t             
