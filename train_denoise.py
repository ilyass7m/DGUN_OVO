"""
Training script for DGUNet Gaussian denoising.
Supports training with synthetic Gaussian noise on clean image datasets
(BSD400, DIV2K, WED, etc.) at configurable noise levels.

Usage:
    python train_denoise.py --sigma 25 --train_dir ./Datasets/train --val_dir ./Datasets/BSD68
"""

import os
import argparse
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from dataset_denoise import GaussianDenoiseTrainDataset, GaussianDenoiseTestDataset
from DGUNet import DGUNet
import losses
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='DGUNet Gaussian Denoising Training')
    # Data
    parser.add_argument('--train_dir', type=str, required=True, help='Path to clean training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to clean validation images (e.g. BSD68)')
    parser.add_argument('--sigma', type=int, default=25, help='Gaussian noise level (15, 25, or 50)')
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--num_workers', type=int, default=4)
    # Model
    parser.add_argument('--n_feat', type=int, default=80)
    parser.add_argument('--depth', type=int, default=5, help='Number of intermediate unfolding stages')
    parser.add_argument('--use_edge_loss', action='store_true', help='Add edge loss term')
    parser.add_argument('--edge_loss_weight', type=float, default=0.05)
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    # GPU
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def train():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = True

    # Directories
    save_dir = os.path.join(args.save_dir, f'denoise_sigma{args.sigma}')
    utils.mkdir(save_dir)

    # Model
    model = DGUNet(n_feat=args.n_feat, depth=args.depth)
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    start_epoch = 1
    best_psnr = 0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume)
        # Handle DataParallel state_dict
        from collections import OrderedDict
        state_dict = ckpt['state_dict']
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_sd[name] = v
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(new_sd)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0)
        print(f'Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}')
        for _ in range(start_epoch - 1):
            scheduler.step()

    # Loss
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss() if args.use_edge_loss else None

    # Data
    train_dataset = GaussianDenoiseTrainDataset(args.train_dir, patch_size=args.patch_size, sigma=args.sigma)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)

    val_dataset = GaussianDenoiseTestDataset(args.val_dir, sigma=args.sigma)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print(f'Training: {len(train_dataset)} images | Validation: {len(val_dataset)} images')
    print(f'Sigma: {args.sigma} | Epochs: {args.epochs} | Batch: {args.batch_size} | Patch: {args.patch_size}')
    print(f'Unfolding stages: {args.depth + 2} (1 first + {args.depth} middle + 1 last)')

    mixup = utils.MixUp_AUG()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for i, (clean, noisy) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            clean = clean.cuda()
            noisy = noisy.cuda()

            # MixUp augmentation after epoch 5
            if epoch > 5:
                clean, noisy = mixup.aug(clean, noisy)

            for p in model.parameters():
                p.grad = None

            restored_stages = model(noisy)

            # Multi-stage loss
            loss = sum(criterion_char(torch.clamp(stage, 0, 1), clean) for stage in restored_stages)
            if criterion_edge is not None:
                loss += args.edge_loss_weight * sum(
                    criterion_edge(torch.clamp(stage, 0, 1), clean) for stage in restored_stages
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f} | Time: {time.time()-t0:.1f}s')

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            psnrs = []
            with torch.no_grad():
                for clean, noisy, fname in val_loader:
                    clean, noisy = clean.cuda(), noisy.cuda()
                    restored = model(noisy)[0]
                    restored = torch.clamp(restored, 0, 1)
                    for r, t in zip(restored, clean):
                        psnrs.append(utils.torchPSNR(r, t).item())

            avg_psnr = np.mean(psnrs)
            print(f'  Validation PSNR: {avg_psnr:.2f} dB')

            is_best = avg_psnr > best_psnr
            if is_best:
                best_psnr = avg_psnr

            base_model = model.module if isinstance(model, nn.DataParallel) else model
            state = {
                'epoch': epoch,
                'state_dict': base_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'sigma': args.sigma,
            }
            torch.save(state, os.path.join(save_dir, 'model_latest.pth'))
            if is_best:
                torch.save(state, os.path.join(save_dir, 'model_best.pth'))
                print(f'  -> New best model saved (PSNR: {best_psnr:.2f})')

    print(f'\nTraining complete. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    train()
