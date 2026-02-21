"""
Stage-by-Stage Visualization for DGUNet

This script visualizes the output at each unfolding stage of DGUNet,
demonstrating how the optimization algorithm iteratively refines the image.

Usage:
    # With synthetic noise (for testing on your own clean images)
    python visualize_stages.py --image path/to/clean_image.jpg --sigma 25 --weights checkpoints/model_best.pth

    # With pre-noised image (if you already have a noisy image)
    python visualize_stages.py --noisy_image path/to/noisy.jpg --clean_image path/to/clean.jpg --weights checkpoints/model_best.pth

    # Quick test on a folder of images
    python visualize_stages.py --image_dir path/to/images/ --sigma 25 --weights checkpoints/model_best.pth
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from skimage.metrics import structural_similarity as compare_ssim
from glob import glob

from DGUNet import DGUNet
from DGUNet_denoise import DGUNet_Denoise


def load_image(path, max_size=512):
    """Load and preprocess an image."""
    img = Image.open(path).convert('RGB')

    # Resize if too large (to fit in memory)
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Ensure dimensions are multiples of 16 (for U-Net)
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))

    return np.array(img).astype(np.float32) / 255.0


def add_gaussian_noise(clean_img, sigma):
    """Add Gaussian noise to a clean image."""
    noise = np.random.randn(*clean_img.shape).astype(np.float32) * (sigma / 255.0)
    noisy = clean_img + noise
    return np.clip(noisy, 0, 1)


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    return compare_ssim(img1, img2, data_range=1.0, channel_axis=2)


def load_model(weights_path, device, n_feat=80, depth=5, known_gradient=False):
    """Load a trained DGUNet model."""
    if known_gradient:
        model = DGUNet_Denoise(n_feat=n_feat, scale_unetfeats=48, depth=depth, known_gradient=True)
    else:
        model = DGUNet(n_feat=n_feat, scale_unetfeats=48, depth=depth)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model


def run_inference(model, noisy_img, device):
    """
    Run inference and return all stage outputs.

    Returns:
        list of numpy arrays: Stage outputs from stage 1 to stage 7
    """
    # Prepare input
    noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model returns [stage7, stage6, ..., stage1]
        outputs = model(noisy_tensor)

    # Convert to numpy and reverse to get [stage1, stage2, ..., stage7]
    stage_outputs = []
    for out in reversed(outputs):
        out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_np = np.clip(out_np, 0, 1)
        stage_outputs.append(out_np)

    return stage_outputs


def visualize_stages(clean_img, noisy_img, stage_outputs, sigma, save_path=None, title_prefix=""):
    """
    Create a visualization showing the progression through stages.
    """
    n_stages = len(stage_outputs)

    # Compute metrics for each stage
    psnrs = [compute_psnr(clean_img, out) for out in stage_outputs]
    ssims = [compute_ssim(clean_img, out) for out in stage_outputs]

    # Also compute metrics for noisy input
    noisy_psnr = compute_psnr(clean_img, noisy_img)
    noisy_ssim = compute_ssim(clean_img, noisy_img)

    # Create figure
    n_cols = min(4, n_stages + 2)  # +2 for clean and noisy
    n_rows = (n_stages + 2 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    # Plot clean image
    axes[0].imshow(clean_img)
    axes[0].set_title(f'Clean (Ground Truth)', fontsize=10)
    axes[0].axis('off')

    # Plot noisy image
    axes[1].imshow(noisy_img)
    axes[1].set_title(f'Noisy (σ={sigma})\nPSNR: {noisy_psnr:.2f} dB', fontsize=10)
    axes[1].axis('off')

    # Plot each stage output
    for i, (out, psnr, ssim) in enumerate(zip(stage_outputs, psnrs, ssims)):
        ax = axes[i + 2]
        ax.imshow(out)
        ax.set_title(f'Stage {i + 1}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}', fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for i in range(n_stages + 2, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{title_prefix}DGUNet Stage-by-Stage Denoising (σ={sigma})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    return psnrs, ssims


def visualize_convergence(psnrs, ssims, sigma, save_path=None):
    """Plot PSNR and SSIM convergence across stages."""
    stages = list(range(1, len(psnrs) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # PSNR plot
    ax1.plot(stages, psnrs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title(f'PSNR Convergence (σ={sigma})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stages)

    # Annotate improvement
    improvement = psnrs[-1] - psnrs[0]
    ax1.annotate(f'+{improvement:.2f} dB', xy=(stages[-1], psnrs[-1]),
                 xytext=(stages[-1] - 1, psnrs[-1] + 0.5),
                 fontsize=10, color='green')

    # SSIM plot
    ax2.plot(stages, ssims, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title(f'SSIM Convergence (σ={sigma})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(stages)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")

    plt.show()


def print_stage_metrics(psnrs, ssims, noisy_psnr, noisy_ssim):
    """Print a table of metrics for each stage."""
    print("\n" + "=" * 60)
    print("Stage-by-Stage Metrics")
    print("=" * 60)
    print(f"{'Stage':<10} {'PSNR (dB)':<15} {'SSIM':<15} {'ΔPSNR':<15}")
    print("-" * 60)
    print(f"{'Noisy':<10} {noisy_psnr:<15.2f} {noisy_ssim:<15.4f} {'-':<15}")

    for i, (psnr, ssim) in enumerate(zip(psnrs, ssims)):
        delta = psnr - noisy_psnr
        print(f"{'Stage ' + str(i+1):<10} {psnr:<15.2f} {ssim:<15.4f} {'+' + f'{delta:.2f}':<15}")

    print("-" * 60)
    total_improvement = psnrs[-1] - noisy_psnr
    print(f"{'Total':<10} {'':<15} {'':<15} {'+' + f'{total_improvement:.2f}':<15}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='DGUNet Stage-by-Stage Visualization')

    # Input options
    parser.add_argument('--image', type=str, help='Path to a clean image (noise will be added)')
    parser.add_argument('--noisy_image', type=str, help='Path to a pre-noised image')
    parser.add_argument('--clean_image', type=str, help='Path to clean reference (required with --noisy_image)')
    parser.add_argument('--image_dir', type=str, help='Directory of clean images to process')

    # Noise settings
    parser.add_argument('--sigma', type=int, default=25, help='Noise level for synthetic noise (default: 25)')

    # Model settings
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--n_feat', type=int, default=80, help='Feature channels (default: 80)')
    parser.add_argument('--depth', type=int, default=5, help='Unfolding depth (default: 5)')
    parser.add_argument('--known_gradient', action='store_true', help='Use known gradient model')

    # Output settings
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Directory to save outputs')
    parser.add_argument('--max_size', type=int, default=512, help='Max image dimension (default: 512)')
    parser.add_argument('--no_display', action='store_true', help='Do not display plots (only save)')

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.noisy_image and not args.image_dir:
        parser.error("Must provide --image, --noisy_image, or --image_dir")

    if args.noisy_image and not args.clean_image:
        parser.error("--clean_image is required when using --noisy_image")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights, device, args.n_feat, args.depth, args.known_gradient)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Collect images to process
    images_to_process = []

    if args.image:
        images_to_process.append(('single', args.image, None))

    if args.noisy_image:
        images_to_process.append(('paired', args.clean_image, args.noisy_image))

    if args.image_dir:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in extensions:
            images_to_process.extend([('single', p, None) for p in glob(os.path.join(args.image_dir, ext))])

    print(f"Processing {len(images_to_process)} image(s)...")

    # Process each image
    all_psnrs = []
    all_ssims = []

    for idx, (mode, img_path, noisy_path) in enumerate(images_to_process):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{idx + 1}/{len(images_to_process)}] Processing: {img_name}")

        # Load images
        clean_img = load_image(img_path, args.max_size)

        if mode == 'paired':
            noisy_img = load_image(noisy_path, args.max_size)
            # Ensure same size
            h, w = clean_img.shape[:2]
            noisy_img = noisy_img[:h, :w, :]
        else:
            # Add synthetic noise
            noisy_img = add_gaussian_noise(clean_img, args.sigma)

        # Run inference
        stage_outputs = run_inference(model, noisy_img, device)

        # Compute metrics
        noisy_psnr = compute_psnr(clean_img, noisy_img)
        noisy_ssim = compute_ssim(clean_img, noisy_img)

        # Visualize
        save_path = os.path.join(args.save_dir, f'{img_name}_stages.png')
        psnrs, ssims = visualize_stages(
            clean_img, noisy_img, stage_outputs, args.sigma,
            save_path=save_path, title_prefix=f"{img_name}: "
        )

        # Print metrics
        print_stage_metrics(psnrs, ssims, noisy_psnr, noisy_ssim)

        # Save convergence plot
        conv_path = os.path.join(args.save_dir, f'{img_name}_convergence.png')
        visualize_convergence(psnrs, ssims, args.sigma, save_path=conv_path)

        all_psnrs.append(psnrs)
        all_ssims.append(ssims)

    # Summary statistics if multiple images
    if len(images_to_process) > 1:
        avg_psnrs = np.mean(all_psnrs, axis=0)
        avg_ssims = np.mean(all_ssims, axis=0)

        print("\n" + "=" * 60)
        print("AVERAGE METRICS ACROSS ALL IMAGES")
        print("=" * 60)
        for i, (psnr, ssim) in enumerate(zip(avg_psnrs, avg_ssims)):
            print(f"Stage {i + 1}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
