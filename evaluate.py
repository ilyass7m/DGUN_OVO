"""
Evaluation script: compute PSNR and SSIM on test sets.

Usage:
    python evaluate.py --weights checkpoints/denoise_sigma25/model_best.pth \
                       --test_dir ./Datasets/BSD68 --sigma 25
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from DGUNet import DGUNet
from dataset_denoise import GaussianDenoiseTestDataset


def load_model(weights_path, n_feat=80, depth=5):
    model = DGUNet(n_feat=n_feat, depth=depth)
    ckpt = torch.load(weights_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_sd[name] = v
    model.load_state_dict(new_sd)
    model.cuda().eval()
    return model


def evaluate(model, test_dir, sigma):
    dataset = GaussianDenoiseTestDataset(test_dir, sigma=sigma)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnrs, ssims = [], []
    with torch.no_grad():
        for clean, noisy, fname in loader:
            clean, noisy = clean.cuda(), noisy.cuda()
            restored = model(noisy)[0]
            restored = torch.clamp(restored, 0, 1)

            # Convert to numpy HWC uint8 for SSIM
            res_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            cln_np = clean[0].cpu().numpy().transpose(1, 2, 0)

            p = compare_psnr(cln_np, res_np, data_range=1.0)
            s = compare_ssim(cln_np, res_np, data_range=1.0, channel_axis=2)
            psnrs.append(p)
            ssims.append(s)
            print(f'  {fname[0]}: PSNR={p:.2f} SSIM={s:.4f}')

    return np.mean(psnrs), np.mean(ssims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--sigma', type=int, default=25)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = load_model(args.weights)
    avg_psnr, avg_ssim = evaluate(model, args.test_dir, args.sigma)
    print(f'\n==> {os.path.basename(args.test_dir)} | sigma={args.sigma}')
    print(f'    PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}')
