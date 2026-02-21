"""
Training script for DGUNet Denoising.

Supports three dataset modes:
  1. synthetic: Clean images + synthetic Gaussian noise (DIV2K, BSD400)
  2. paired: Preprocessed noisy/clean pairs (input/ and target/ subdirs)
  3. sidd: Native SIDD structure (GT_SRGB_*.PNG and NOISY_SRGB_*.PNG)

Usage (synthetic noise - DIV2K):
    python train.py --dataset_mode synthetic --train_dir ./Datasets/DIV2K_train_HR --val_dir ./Datasets/DIV2K_valid_HR --sigma 25 --wandb

Usage (SIDD with proper train/val split - 140 train, 20 val):
    python train.py --dataset_mode sidd --train_dir ./Datasets/SIDD_Small_sRGB_Only \
                    --val_dir ./Datasets/SIDD_Small_sRGB_Only --sidd_split --wandb --name dgunet_sidd

Usage (fast training with AMP - recommended for 12GB VRAM):
    python train.py --dataset_mode synthetic --train_dir ./Datasets/DIV2K_train_HR \
                    --val_dir ./Datasets/DIV2K_valid_HR --sigma 25 --amp --batch_size 16 --wandb

Hyperparameters (matching paper):
    - lr=1e-4, warmup_epochs=3, extended cosine annealing
    - batch_size=2, accum_steps=2 -> effective batch=4 (paper uses 16)

Performance options:
    --amp       Enable automatic mixed precision (2-3x speedup, ~40% less VRAM)
    --compile   Use torch.compile() for PyTorch 2.0+ (10-30% speedup)

Ablation studies:
    --no_isff           Disable Inter-Stage Feature Fusion (ISFF) module
    --known_gradient    Use analytical gradient instead of learned phi/phiT

Example (ISFF ablation):
    python train.py --dataset_mode synthetic --train_dir ./Datasets/DIV2K_train_HR \
                    --val_dir ./Datasets/DIV2K_valid_HR --sigma 25 --no_isff --name dgunet_no_isff --wandb
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim

from DGUNet import DGUNet
from DGUNet_denoise import DGUNet_Denoise
from DGUNet_ablation import DGUNet_Ablation
from dataset_denoise import (
    GaussianDenoiseTrainDataset, GaussianDenoiseTestDataset,
    PairedDenoiseDataset, PairedDenoiseTestDataset,
    SIDDTrainDataset, SIDDTestDataset
)
import losses
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='DGUNet Gaussian Denoising Training')

    # Data
    parser.add_argument('--dataset_mode', type=str, default='synthetic', choices=['synthetic', 'paired', 'sidd'],
                        help='Dataset mode: synthetic (Gaussian), paired (input/target dirs), sidd (native SIDD structure)')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data')
    parser.add_argument('--sigma', type=int, default=25, help='Noise level for synthetic mode (default: 25)')
    parser.add_argument('--sidd_split', action='store_true',
                        help='For SIDD mode: split into 140 train / 20 val (uses same dir for both)')

    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 2)')
    parser.add_argument('--patch_size', type=int, default=128, help='Training patch size (default: 128)')
    parser.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--patches_per_image', type=int, default=1, help='Patches per image per epoch (default: 1)')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs (default: 80)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 1e-4, same as paper)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs (default: 3)')

    # Validation
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs (default: 5)')
    parser.add_argument('--val_crop', type=int, default=256, help='Validation center crop size (default: 256)')
    parser.add_argument('--val_mode', type=str, default=None, choices=['synthetic', 'paired', 'sidd'],
                        help='Validation dataset mode (default: same as dataset_mode). Use for cross-dataset validation.')

    # Model
    parser.add_argument('--n_feat', type=int, default=80, help='Feature channels (default: 80)')
    parser.add_argument('--depth', type=int, default=5, help='Unfolding depth (default: 5, gives 7 stages)')
    parser.add_argument('--known_gradient', action='store_true',
                        help='Use analytical gradient (x-y) instead of learned phi/phiT. Only for denoising (H=I).')
    parser.add_argument('--no_isff', action='store_true',
                        help='Disable Inter-Stage Feature Fusion (ISFF) for ablation study')

    # Loss
    parser.add_argument('--edge_loss', action='store_true', help='Use edge loss')
    parser.add_argument('--edge_weight', type=float, default=0.05, help='Edge loss weight (default: 0.05)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=20, help='Save checkpoint every N epochs (default: 10)')
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

    # Performance optimizations
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (AMP) for faster training')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for PyTorch 2.0+ speedup')

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
    if args.known_gradient:
        logger.info("Using DGUNet_Denoise with KNOWN gradient (H=I, analytical)")
        model = DGUNet_Denoise(n_feat=args.n_feat, scale_unetfeats=48, depth=args.depth,
                               known_gradient=True).to(device)
    elif args.no_isff:
        logger.info("Using DGUNet WITHOUT Inter-Stage Feature Fusion (ISFF ablation)")
        model = DGUNet_Ablation(n_feat=args.n_feat, scale_unetfeats=48, depth=args.depth,
                                use_isff=False).to(device)
    else:
        logger.info("Using DGUNet with LEARNED gradient (phi/phiT ResBlocks)")
        model = DGUNet(n_feat=args.n_feat, scale_unetfeats=48, depth=args.depth).to(device)

    # torch.compile for PyTorch 2.0+ speedup
    if args.compile:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled successfully")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # AMP scaler for mixed precision training
    scaler = GradScaler(enabled=args.amp)
    if args.amp:
        logger.info("Automatic Mixed Precision (AMP) enabled")

    if args.wandb:
        wandb.config.update({"n_params": n_params})

    # Optimizer & Scheduler (with warmup like original paper)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

    
    T_max = args.epochs - args.warmup_epochs    #+ 40
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=args.lr_min
    )

    # Warmup scheduler: linear warmup for first warmup_epochs
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Combined: use warmup for first N epochs, then cosine
    use_warmup = True

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
        # Advance schedulers to correct position
        for e in range(1, start_epoch):
            if e <= args.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
        logger.info(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}")

    # Loss
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss() if args.edge_loss else None

    # Training dataset - select based on dataset_mode
    if args.dataset_mode == 'synthetic':
        logger.info(f"Train mode: SYNTHETIC (sigma={args.sigma})")
        train_dataset = GaussianDenoiseTrainDataset(
            args.train_dir, patch_size=args.patch_size,
            sigma=args.sigma, patches_per_image=args.patches_per_image
        )
    elif args.dataset_mode == 'sidd':
        split_train = 'train' if args.sidd_split else None
        logger.info(f"Train mode: SIDD (split={split_train}, {140 if args.sidd_split else 160} scenes)")
        train_dataset = SIDDTrainDataset(
            args.train_dir, patch_size=args.patch_size, augment=True, split=split_train
        )
    else:  # paired
        logger.info("Train mode: PAIRED (input/target directory structure)")
        train_dataset = PairedDenoiseDataset(
            args.train_dir, patch_size=args.patch_size, augment=True
        )

    # Validation dataset - use val_mode if specified, otherwise same as dataset_mode
    val_mode = args.val_mode if args.val_mode else args.dataset_mode
    if val_mode == 'synthetic':
        logger.info(f"Val mode: SYNTHETIC (sigma={args.sigma}) on {args.val_dir}")
        val_dataset = GaussianDenoiseTestDataset(
            args.val_dir, sigma=args.sigma, center_crop=args.val_crop
        )
    elif val_mode == 'sidd':
        split_val = 'val' if args.sidd_split else None
        logger.info(f"Val mode: SIDD (split={split_val}, {20 if args.sidd_split else 160} scenes)")
        val_dataset = SIDDTestDataset(
            args.val_dir, center_crop=args.val_crop, split=split_val
        )
    else:  # paired
        logger.info(f"Val mode: PAIRED on {args.val_dir}")
        val_dataset = PairedDenoiseTestDataset(
            args.val_dir, center_crop=args.val_crop
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=3 if args.workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True
    )

    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} images")
    logger.info(f"Iterations per epoch: {len(train_loader)}")

    mixup = utils.MixUp_AUG()

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (clean, noisy) in enumerate(pbar):
            # Non-blocking transfers for better GPU utilization
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)

            if epoch > 5:
                clean, noisy = mixup.aug(clean, noisy)

            # Forward pass with automatic mixed precision
            with autocast(enabled=args.amp):
                restored = model(noisy)
                loss = sum(criterion_char(torch.clamp(s, 0, 1), clean) for s in restored)
                if criterion_edge:
                    loss += args.edge_weight * sum(criterion_edge(torch.clamp(s, 0, 1), clean) for s in restored)

            # Normalize loss for gradient accumulation
            loss = loss / args.accum_steps

            # Backward pass with gradient scaling for AMP
            scaler.scale(loss).backward()

            # Gradient accumulation: only step every accum_steps
            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # Faster than manual p.grad = None
                # Periodically clear cache to prevent fragmentation
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            batch_loss = loss.item() * args.accum_steps  # Report actual loss
            epoch_loss += batch_loss
            global_step += 1

            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            # Log to wandb every 10 steps
            if args.wandb and global_step % 10 == 0:
                lr_now = warmup_scheduler.get_last_lr()[0] if epoch <= args.warmup_epochs else cosine_scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': batch_loss,
                    'train/lr': lr_now,
                    'train/epoch': epoch,
                    'train/step': global_step
                }, step=global_step)

        # Update learning rate (warmup then cosine)
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step()
            current_lr = cosine_scheduler.get_last_lr()[0]

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Log epoch metrics to wandb
        if args.wandb:
            wandb.log({
                'epoch/train_loss': avg_loss,
                'epoch/lr': current_lr,
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
                    clean = clean.to(device, non_blocking=True)
                    noisy = noisy.to(device, non_blocking=True)
                    with autocast(enabled=args.amp):
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
