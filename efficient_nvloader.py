import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from torchvision.io import write_video
from hnerv_utils import dequant_tensor


def load_quantized_video_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    version = checkpoint.get('format_version', 1)
    if version not in (1, 2):
        raise ValueError(
            f"Unsupported quantized checkpoint format_version={version}; supported versions are 1 and 2.")
    if 'embed' not in checkpoint or 'model' not in checkpoint:
        raise ValueError("Quantized checkpoint must contain 'embed' and 'model' records.")
    embedding = dequant_tensor(checkpoint['embed'])
    model_state = {key: dequant_tensor(record) for key, record in checkpoint['model'].items()}
    if version == 2:
        model_state.update(checkpoint.get('buffers', {}))
    return embedding, model_state, checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default='checkpoints/img_decoder.pth', help='path for video decoder',)
    parser.add_argument('--ckt', type=str, default='checkpoints/quant_vid.pth', help='path for video checkpoint',) #
    parser.add_argument('--dump_dir', type=str, default='visualize/bunny_1.5M_E300', help='path for video checkpoint',) #
    parser.add_argument('--frames', type=int, default=16, help='video frames for output',) #

    args = parser.parse_args()

    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Load video checkpoints and dequant them
    vid_embed, dequant_ckt, _ = load_quantized_video_checkpoint(args.ckt)
    vid_embed = vid_embed.to(device)
    dequant_ckt = {key: value.to(device) for key, value in dequant_ckt.items()}
    img_decoder = torch.jit.load(args.decoder, map_location='cpu').to(device)
    img_decoder.load_state_dict(dequant_ckt)

    # Select frame indexs and reconstruct them
    frame_step = vid_embed.size(0) // args.frames
    frame_idx = np.arange(0, vid_embed.size(0), frame_step)[:args.frames]
    img_out = img_decoder(vid_embed[frame_idx]).cpu()

    # Dump video and frames
    out_vid = os.path.join(args.dump_dir, 'nvloader_out.mp4')
    write_video(out_vid, img_out.permute(0,2,3,1) * 255., fps=args.frames/4, options={'crf':'10'})
    for idx in range(args.frames):
        out_img = os.path.join(args.dump_dir, f'frame{idx}_out.png')
        save_image(img_out[idx], out_img)
    print(f'dumped video to {out_vid}')


if __name__ == '__main__':
    main()
