"""
Training script for DGUNet Gaussian Denoising.

Usage:
    python train.py --train_dir ./Datasets/DIV2K_train_HR --val_dir ./Datasets/DIV2K_valid_HR --sigma 25

With wandb logging:
    python train.py --train_dir ./Datasets/DIV2K_train_HR --val_dir ./Datasets/DIV2K_valid_HR --sigma 25 --wandb

Memory-optimized settings (default: batch_size=2, accum_steps=2 -> effective batch=4)
For 4GB GPU, use: --batch_size 1 --accum_steps 4
"""

import os
import sys
import argparse
import random
import time
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim

from DGUNet import DGUNet
from dataset_denoise import GaussianDenoiseTrainDataset, GaussianDenoiseTestDataset
import losses
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='DGUNet Gaussian Denoising Training')

    # Data
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation images')
    parser.add_argument('--sigma', type=int, default=25, help='Noise level (default: 25)')

    # Training
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--patch_size', type=int, default=128, help='Training patch size (default: 128)')
    parser.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--patches_per_image', type=int, default=1, help='Patches per image per epoch (default: 1)')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs (default: 80)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate (default: 1e-6)')

    # Validation
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs (default: 5)')
    parser.add_argument('--val_crop', type=int, default=256, help='Validation center crop size (default: 256)')

    # Model
    parser.add_argument('--n_feat', type=int, default=80, help='Feature channels (default: 80)')
    parser.add_argument('--depth', type=int, default=5, help='Unfolding depth (default: 5, gives 7 stages)')

    # Loss
    parser.add_argument('--edge_loss', action='store_true', help='Use edge loss')
    parser.add_argument('--edge_weight', type=float, default=0.05, help='Edge loss weight (default: 0.05)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--name', type=str, default='dgunet', help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='DGUNet-Denoising', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team)')

    # Hardware
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--workers', type=int, default=6, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    return parser.parse_args()


def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def compute_ssim(restored, clean):
    """Compute SSIM between restored and clean images (numpy arrays in HWC format, range [0,1])."""
    return compare_ssim(clean, restored, data_range=1.0, channel_axis=2)


def main():
    args = parse_args()

    # GPU setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # Checkpoint directory
    ckpt_dir = os.path.join(args.save_dir, f'{args.name}_sigma{args.sigma}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logging
    logger = setup_logging(ckpt_dir)
    logger.info(f"Device: {device}")
    logger.info(f"Save dir: {ckpt_dir}")
    logger.info(f"Args: {vars(args)}")

    # Wandb initialization
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.name}",
            config=vars(args),
            dir=ckpt_dir
        )
        logger.info("Wandb logging enabled")

    # Model
    model = DGUNet(n_feat=args.n_feat, scale_unetfeats=48, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    if args.wandb:
        wandb.config.update({"n_params": n_params})

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    start_epoch = 1
    best_psnr = 0
    global_step = 0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['state_dict']
        new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_sd)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0)
        global_step = ckpt.get('global_step', 0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        logger.info(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}")

    # Loss
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss() if args.edge_loss else None

    # Data
    train_dataset = GaussianDenoiseTrainDataset(
        args.train_dir, patch_size=args.patch_size,
        sigma=args.sigma, patches_per_image=args.patches_per_image
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True
    )

    val_dataset = GaussianDenoiseTestDataset(
        args.val_dir, sigma=args.sigma, center_crop=args.val_crop
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    logger.info(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} images")
    logger.info(f"Iterations per epoch: {len(train_loader)}")

    mixup = utils.MixUp_AUG()

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (clean, noisy) in enumerate(pbar):
            clean, noisy = clean.to(device), noisy.to(device)

            if epoch > 5:
                clean, noisy = mixup.aug(clean, noisy)

            restored = model(noisy)
            loss = sum(criterion_char(torch.clamp(s, 0, 1), clean) for s in restored)
            if criterion_edge:
                loss += args.edge_weight * sum(criterion_edge(torch.clamp(s, 0, 1), clean) for s in restored)

            # Normalize loss for gradient accumulation
            loss = loss / args.accum_steps
            loss.backward()

            # Gradient accumulation: only step every accum_steps
            if (batch_idx + 1) % args.accum_steps == 0:
                optimizer.step()
                for p in model.parameters():
                    p.grad = None
                # Periodically clear cache to prevent fragmentation
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            batch_loss = loss.item() * args.accum_steps  # Report actual loss
            epoch_loss += batch_loss
            global_step += 1

            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            # Log to wandb every 10 steps
            if args.wandb and global_step % 10 == 0:
                wandb.log({
                    'train/loss': batch_loss,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': global_step
                }, step=global_step)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Log epoch metrics to wandb
        if args.wandb:
            wandb.log({
                'epoch/train_loss': avg_loss,
                'epoch/lr': scheduler.get_last_lr()[0],
                'epoch/time': epoch_time
            }, step=global_step)

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            torch.cuda.empty_cache()  # Clear memory before validation
            psnrs = []
            ssims = []

            with torch.no_grad():
                for clean, noisy, fname in val_loader:
                    clean, noisy = clean.to(device), noisy.to(device)
                    restored = torch.clamp(model(noisy)[0], 0, 1)

                    for r, t in zip(restored, clean):
                        # PSNR
                        psnrs.append(utils.torchPSNR(r, t).item())

                        # SSIM (convert to numpy HWC)
                        r_np = r.cpu().numpy().transpose(1, 2, 0)
                        t_np = t.cpu().numpy().transpose(1, 2, 0)
                        ssims.append(compute_ssim(r_np, t_np))

            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            torch.cuda.empty_cache()  # Clear memory after validation

            logger.info(f'Epoch {epoch} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | Time: {epoch_time:.1f}s')

            # Log validation metrics to wandb
            if args.wandb:
                wandb.log({
                    'val/psnr': avg_psnr,
                    'val/ssim': avg_ssim,
                    'val/epoch': epoch
                }, step=global_step)

            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': avg_ssim,
                    'global_step': global_step
                }, os.path.join(ckpt_dir, 'model_best.pth'))
                logger.info(f'  -> Best model saved (PSNR: {best_psnr:.2f}, SSIM: {avg_ssim:.4f})')

                if args.wandb:
                    wandb.run.summary['best_psnr'] = best_psnr
                    wandb.run.summary['best_ssim'] = avg_ssim
                    wandb.run.summary['best_epoch'] = epoch
        else:
            logger.info(f'Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s')

        # Save checkpoint every save_every epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'global_step': global_step
            }, os.path.join(ckpt_dir, f'model_epoch_{epoch}.pth'))
            logger.info(f'  -> Checkpoint saved: model_epoch_{epoch}.pth')

        # Save latest (every epoch)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'global_step': global_step
        }, os.path.join(ckpt_dir, 'model_latest.pth'))

    logger.info(f'Training complete. Best PSNR: {best_psnr:.2f} dB')

    # Finish wandb run
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
