"""
ISFF Ablation Comparison Script

This script compares DGUNet with and without Inter-Stage Feature Fusion (ISFF)
to demonstrate the impact of this module on denoising performance.

ISFF Components:
    1. mergeblock: Subspace projection merging features between consecutive stages
    2. CSFF: Cross-Stage Feature Fusion in encoder using previous stage's features

Expected Results:
    - Model WITH ISFF should show better detail preservation
    - Model WITHOUT ISFF may show more artifacts or loss of fine details
    - PSNR/SSIM should be higher with ISFF enabled

Usage:
    python compare_isff_ablation.py \
        --weights_isff checkpoints/dgunet_with_isff/model_best.pth \
        --weights_no_isff checkpoints/dgunet_no_isff/model_best.pth \
        --test_dir ./Datasets/DIV2K_valid_HR \
        --sigma 25
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
from tqdm import tqdm

from DGUNet import DGUNet
from DGUNet_ablation import DGUNet_Ablation


def load_image(path, max_size=512):
    """Load and preprocess an image."""
    img = Image.open(path).convert('RGB')

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Ensure dimensions are multiples of 16
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))

    return np.array(img).astype(np.float32) / 255.0


def add_gaussian_noise(clean_img, sigma, seed=None):
    """Add Gaussian noise with optional seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
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


def load_model(weights_path, device, n_feat=80, depth=5, use_isff=True):
    """Load a DGUNet model."""
    if use_isff:
        model = DGUNet(n_feat=n_feat, scale_unetfeats=48, depth=depth)
    else:
        model = DGUNet_Ablation(n_feat=n_feat, scale_unetfeats=48, depth=depth, use_isff=False)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model


def denoise_image(model, noisy_img, device):
    """Run denoising inference."""
    noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(noisy_tensor)
        denoised = outputs[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)

    return np.clip(denoised, 0, 1)


def get_all_stage_outputs(model, noisy_img, device):
    """Get outputs from all stages."""
    noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(noisy_tensor)

    # Reverse to get stage 1 -> stage 7
    stage_outputs = []
    for out in reversed(outputs):
        out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        stage_outputs.append(np.clip(out_np, 0, 1))

    return stage_outputs


def create_comparison_figure(clean, noisy, denoised_isff, denoised_no_isff,
                             metrics_isff, metrics_no_isff, sigma, img_name):
    """Create side-by-side comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Full images
    axes[0, 0].imshow(clean)
    axes[0, 0].set_title('Ground Truth', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised_isff)
    axes[0, 1].set_title(f'WITH ISFF\nPSNR: {metrics_isff["psnr"]:.2f} dB | SSIM: {metrics_isff["ssim"]:.4f}',
                         fontsize=12, color='green')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised_no_isff)
    axes[0, 2].set_title(f'WITHOUT ISFF\nPSNR: {metrics_no_isff["psnr"]:.2f} dB | SSIM: {metrics_no_isff["ssim"]:.4f}',
                         fontsize=12, color='red')
    axes[0, 2].axis('off')

    # Bottom row: Zoomed regions (center crop)
    h, w = clean.shape[:2]
    crop_size = min(h, w) // 3
    y1, x1 = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    y2, x2 = y1 + crop_size, x1 + crop_size

    axes[1, 0].imshow(clean[y1:y2, x1:x2])
    axes[1, 0].set_title('Ground Truth (Zoomed)', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_isff[y1:y2, x1:x2])
    axes[1, 1].set_title('WITH ISFF (Zoomed)', fontsize=12, color='green')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_no_isff[y1:y2, x1:x2])
    axes[1, 2].set_title('WITHOUT ISFF (Zoomed)', fontsize=12, color='red')
    axes[1, 2].axis('off')

    # Add difference indicator
    psnr_diff = metrics_isff['psnr'] - metrics_no_isff['psnr']
    ssim_diff = metrics_isff['ssim'] - metrics_no_isff['ssim']

    fig.suptitle(f'{img_name} | σ={sigma} | ISFF Advantage: +{psnr_diff:.2f} dB PSNR, +{ssim_diff:.4f} SSIM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def create_stage_comparison(clean, noisy, stages_isff, stages_no_isff, sigma):
    """Compare stage-by-stage progression for both models."""
    n_stages = len(stages_isff)

    # Compute PSNR at each stage
    psnrs_isff = [compute_psnr(clean, s) for s in stages_isff]
    psnrs_no_isff = [compute_psnr(clean, s) for s in stages_no_isff]

    # Create convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stages = list(range(1, n_stages + 1))

    # PSNR convergence
    ax1.plot(stages, psnrs_isff, 'g-o', linewidth=2, markersize=8, label='WITH ISFF')
    ax1.plot(stages, psnrs_no_isff, 'r-s', linewidth=2, markersize=8, label='WITHOUT ISFF')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title(f'PSNR Convergence Across Stages (σ={sigma})', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stages)

    # Gap visualization
    gaps = [psnrs_isff[i] - psnrs_no_isff[i] for i in range(n_stages)]
    colors = ['green' if g > 0 else 'red' for g in gaps]
    ax2.bar(stages, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('PSNR Difference (dB)', fontsize=12)
    ax2.set_title('ISFF Advantage per Stage\n(Positive = ISFF better)', fontsize=14)
    ax2.set_xticks(stages)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, psnrs_isff, psnrs_no_isff


def create_error_map_comparison(clean, denoised_isff, denoised_no_isff, img_name):
    """Create error map visualization."""
    # Compute absolute errors
    error_isff = np.abs(clean - denoised_isff).mean(axis=2)
    error_no_isff = np.abs(clean - denoised_no_isff).mean(axis=2)

    # Find common scale
    vmax = max(error_isff.max(), error_no_isff.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Error map WITH ISFF
    im1 = axes[0].imshow(error_isff, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title(f'Error Map: WITH ISFF\nMean Error: {error_isff.mean():.4f}', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Error map WITHOUT ISFF
    im2 = axes[1].imshow(error_no_isff, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(f'Error Map: WITHOUT ISFF\nMean Error: {error_no_isff.mean():.4f}', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Difference (where ISFF helps more)
    diff = error_no_isff - error_isff  # Positive = ISFF better
    im3 = axes[2].imshow(diff, cmap='RdYlGn', vmin=-vmax/2, vmax=vmax/2)
    axes[2].set_title('Difference (Green = ISFF better)\nRed = No-ISFF better', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    fig.suptitle(f'{img_name} - Error Analysis', fontsize=14)
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare ISFF vs No-ISFF')

    # Model weights
    parser.add_argument('--weights_isff', type=str, required=True,
                        help='Path to model weights WITH ISFF')
    parser.add_argument('--weights_no_isff', type=str, required=True,
                        help='Path to model weights WITHOUT ISFF')

    # Test data
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--sigma', type=int, default=25, help='Noise level')

    # Model config
    parser.add_argument('--n_feat', type=int, default=80, help='Feature channels')
    parser.add_argument('--depth', type=int, default=5, help='Model depth')

    # Output
    parser.add_argument('--save_dir', type=str, default='./isff_comparison', help='Output directory')
    parser.add_argument('--max_images', type=int, default=10, help='Max images to process')
    parser.add_argument('--max_size', type=int, default=512, help='Max image dimension')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    model_isff = load_model(args.weights_isff, device, args.n_feat, args.depth, use_isff=True)
    model_no_isff = load_model(args.weights_no_isff, device, args.n_feat, args.depth, use_isff=False)

    params_isff = sum(p.numel() for p in model_isff.parameters())
    params_no_isff = sum(p.numel() for p in model_no_isff.parameters())
    print(f"Model WITH ISFF:    {params_isff:,} params")
    print(f"Model WITHOUT ISFF: {params_no_isff:,} params")

    # Collect test images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob(os.path.join(args.test_dir, ext)))
        image_paths.extend(glob(os.path.join(args.test_dir, ext.upper())))

    image_paths = image_paths[:args.max_images]
    print(f"\nProcessing {len(image_paths)} images...")

    # Results storage
    all_results = []

    for img_path in tqdm(image_paths, desc="Processing"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load and add noise (same noise for both models)
        clean = load_image(img_path, args.max_size)
        noisy = add_gaussian_noise(clean, args.sigma, seed=42)  # Fixed seed for fair comparison

        # Denoise with both models
        denoised_isff = denoise_image(model_isff, noisy, device)
        denoised_no_isff = denoise_image(model_no_isff, noisy, device)

        # Compute metrics
        metrics_isff = {
            'psnr': compute_psnr(clean, denoised_isff),
            'ssim': compute_ssim(clean, denoised_isff)
        }
        metrics_no_isff = {
            'psnr': compute_psnr(clean, denoised_no_isff),
            'ssim': compute_ssim(clean, denoised_no_isff)
        }

        all_results.append({
            'name': img_name,
            'isff': metrics_isff,
            'no_isff': metrics_no_isff
        })

        # Create comparison figure
        fig = create_comparison_figure(clean, noisy, denoised_isff, denoised_no_isff,
                                       metrics_isff, metrics_no_isff, args.sigma, img_name)
        fig.savefig(os.path.join(args.save_dir, f'{img_name}_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Create error map
        fig_err = create_error_map_comparison(clean, denoised_isff, denoised_no_isff, img_name)
        fig_err.savefig(os.path.join(args.save_dir, f'{img_name}_error_map.png'),
                        dpi=150, bbox_inches='tight')
        plt.close(fig_err)

        # Stage-by-stage comparison (for first image only)
        if img_path == image_paths[0]:
            stages_isff = get_all_stage_outputs(model_isff, noisy, device)
            stages_no_isff = get_all_stage_outputs(model_no_isff, noisy, device)

            fig_stages, psnrs_isff, psnrs_no_isff = create_stage_comparison(
                clean, noisy, stages_isff, stages_no_isff, args.sigma)
            fig_stages.savefig(os.path.join(args.save_dir, 'stage_comparison.png'),
                               dpi=150, bbox_inches='tight')
            plt.close(fig_stages)

    # Print summary
    print("\n" + "=" * 70)
    print("ISFF ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Image':<25} {'ISFF PSNR':<12} {'No-ISFF PSNR':<14} {'Δ PSNR':<10} {'Δ SSIM':<10}")
    print("-" * 70)

    total_psnr_isff = 0
    total_psnr_no_isff = 0
    total_ssim_isff = 0
    total_ssim_no_isff = 0

    for r in all_results:
        delta_psnr = r['isff']['psnr'] - r['no_isff']['psnr']
        delta_ssim = r['isff']['ssim'] - r['no_isff']['ssim']
        print(f"{r['name']:<25} {r['isff']['psnr']:<12.2f} {r['no_isff']['psnr']:<14.2f} "
              f"{'+' if delta_psnr >= 0 else ''}{delta_psnr:<9.2f} "
              f"{'+' if delta_ssim >= 0 else ''}{delta_ssim:<10.4f}")

        total_psnr_isff += r['isff']['psnr']
        total_psnr_no_isff += r['no_isff']['psnr']
        total_ssim_isff += r['isff']['ssim']
        total_ssim_no_isff += r['no_isff']['ssim']

    n = len(all_results)
    avg_psnr_isff = total_psnr_isff / n
    avg_psnr_no_isff = total_psnr_no_isff / n
    avg_ssim_isff = total_ssim_isff / n
    avg_ssim_no_isff = total_ssim_no_isff / n

    print("-" * 70)
    print(f"{'AVERAGE':<25} {avg_psnr_isff:<12.2f} {avg_psnr_no_isff:<14.2f} "
          f"+{avg_psnr_isff - avg_psnr_no_isff:<9.2f} "
          f"+{avg_ssim_isff - avg_ssim_no_isff:<10.4f}")
    print("=" * 70)

    print(f"\nConclusion: ISFF provides +{avg_psnr_isff - avg_psnr_no_isff:.2f} dB average PSNR improvement")
    print(f"Results saved to: {args.save_dir}")

    # Save summary to file
    with open(os.path.join(args.save_dir, 'summary.txt'), 'w') as f:
        f.write("ISFF Ablation Study Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Noise Level: σ = {args.sigma}\n")
        f.write(f"Test Images: {len(all_results)}\n\n")
        f.write(f"Average PSNR WITH ISFF:    {avg_psnr_isff:.2f} dB\n")
        f.write(f"Average PSNR WITHOUT ISFF: {avg_psnr_no_isff:.2f} dB\n")
        f.write(f"ISFF Advantage:            +{avg_psnr_isff - avg_psnr_no_isff:.2f} dB\n\n")
        f.write(f"Average SSIM WITH ISFF:    {avg_ssim_isff:.4f}\n")
        f.write(f"Average SSIM WITHOUT ISFF: {avg_ssim_no_isff:.4f}\n")
        f.write(f"ISFF Advantage:            +{avg_ssim_isff - avg_ssim_no_isff:.4f}\n")


if __name__ == '__main__':
    main()
