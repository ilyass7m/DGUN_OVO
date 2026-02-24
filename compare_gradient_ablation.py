"""
Ablation Study: Known vs Learned Gradient in DGUNet

This script compares two gradient computation strategies:
1. Known Gradient: Uses analytical gradient ∇f(x) = x - y (optimal for denoising)
2. Learned Gradient: Uses ResBlocks φ and φᵀ to approximate the gradient

For Gaussian denoising where H = I (identity), the gradient is exactly known.
The question is: does learning it provide any benefit, or is the analytical form better?

Usage:
    python compare_gradient_ablation.py \
        --known_ckpt ./checkpoints/known_gradient/model_best.pth \
        --learned_ckpt ./checkpoints/learned_gradient/model_best.pth \
        --test_dir ./Datasets/DIV2K_valid_HR \
        --sigma 25
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from PIL import Image

from DGUNet import DGUNet
from DGUNet_denoise import DGUNet_Denoise
from dataset_denoise import GaussianDenoiseTestDataset
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Known vs Learned Gradient Ablation')
    parser.add_argument('--known_ckpt', type=str, required=True,
                        help='Checkpoint for known gradient model')
    parser.add_argument('--learned_ckpt', type=str, required=True,
                        help='Checkpoint for learned gradient model')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test image directory')
    parser.add_argument('--sigma', type=int, default=25,
                        help='Noise level for synthetic noise')
    parser.add_argument('--n_feat', type=int, default=80,
                        help='Feature channels')
    parser.add_argument('--depth', type=int, default=5,
                        help='Unfolding depth')
    parser.add_argument('--save_dir', type=str, default='./results/gradient_ablation',
                        help='Directory to save results')
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP for inference')
    return parser.parse_args()


def load_model(checkpoint_path, n_feat=80, depth=5, known_gradient=False, device='cuda'):
    """Load a DGUNet model with specified gradient mode."""
    model = DGUNet_Denoise(
        n_feat=n_feat,
        scale_unetfeats=48,
        depth=depth,
        known_gradient=known_gradient
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_psnr': checkpoint.get('best_psnr', 'N/A'),
        'best_ssim': checkpoint.get('best_ssim', 'N/A'),
    }

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    info['params'] = n_params

    return model, info


def pad_to_multiple(img, multiple=16):
    """Pad image to be divisible by multiple."""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, (h, w)


def evaluate_model(model, test_loader, device, use_amp=False):
    """Evaluate model and return metrics."""
    model.eval()
    psnrs, ssims = [], []

    with torch.no_grad():
        for clean, noisy, fname in tqdm(test_loader, desc='Evaluating', leave=False):
            clean = clean.to(device)
            noisy = noisy.to(device)

            noisy_pad, (orig_h, orig_w) = pad_to_multiple(noisy, 16)

            if use_amp:
                with autocast(enabled=True):
                    restored = model(noisy_pad)[0]
            else:
                restored = model(noisy_pad)[0]

            restored = restored[:, :, :orig_h, :orig_w]
            restored = torch.clamp(restored, 0, 1)

            res_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            cln_np = clean[0].cpu().numpy().transpose(1, 2, 0)

            psnrs.append(compare_psnr(cln_np, res_np, data_range=1.0))
            ssims.append(compare_ssim(cln_np, res_np, data_range=1.0, channel_axis=2))

    return {
        'psnr': np.mean(psnrs),
        'ssim': np.mean(ssims),
        'psnr_std': np.std(psnrs),
        'ssim_std': np.std(ssims),
    }


def visualize_comparison(known_model, learned_model, test_dataset, device, save_path, use_amp=False):
    """Create visual comparison of the two models."""
    # Select a few test images
    indices = [0, 10, 25] if len(test_dataset) > 25 else [0]

    fig, axes = plt.subplots(len(indices), 5, figsize=(20, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        clean, noisy, fname = test_dataset[idx]
        clean_np = clean.numpy().transpose(1, 2, 0)
        noisy_np = noisy.numpy().transpose(1, 2, 0)

        noisy_t = noisy.unsqueeze(0).to(device)
        noisy_pad, (h, w) = pad_to_multiple(noisy_t, 16)

        with torch.no_grad():
            if use_amp:
                with autocast(enabled=True):
                    known_out = known_model(noisy_pad)[0]
                    learned_out = learned_model(noisy_pad)[0]
            else:
                known_out = known_model(noisy_pad)[0]
                learned_out = learned_model(noisy_pad)[0]

        known_np = torch.clamp(known_out[:, :, :h, :w], 0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        learned_np = torch.clamp(learned_out[:, :, :h, :w], 0, 1)[0].cpu().numpy().transpose(1, 2, 0)

        # Compute metrics
        psnr_known = compare_psnr(clean_np, known_np, data_range=1.0)
        psnr_learned = compare_psnr(clean_np, learned_np, data_range=1.0)
        psnr_noisy = compare_psnr(clean_np, np.clip(noisy_np, 0, 1), data_range=1.0)

        # Compute error maps
        error_known = np.abs(clean_np - known_np).mean(axis=2)
        error_learned = np.abs(clean_np - learned_np).mean(axis=2)

        # Plot
        axes[row, 0].imshow(np.clip(noisy_np, 0, 1))
        axes[row, 0].set_title(f'Noisy\n{psnr_noisy:.2f} dB', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(known_np)
        axes[row, 1].set_title(f'Known Gradient\n{psnr_known:.2f} dB', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(learned_np)
        axes[row, 2].set_title(f'Learned Gradient\n{psnr_learned:.2f} dB', fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(clean_np)
        axes[row, 3].set_title('Ground Truth', fontsize=10)
        axes[row, 3].axis('off')

        # Difference map (known - learned)
        diff = error_known - error_learned
        im = axes[row, 4].imshow(diff, cmap='RdBu', vmin=-0.05, vmax=0.05)
        axes[row, 4].set_title(f'Error Diff\n(red=known worse)', fontsize=10)
        axes[row, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70)
    print("ABLATION STUDY: Known vs Learned Gradient")
    print("="*70)

    # Load models
    print("\nLoading models...")
    known_model, known_info = load_model(
        args.known_ckpt, args.n_feat, args.depth,
        known_gradient=True, device=device
    )
    print(f"Known Gradient Model:")
    print(f"  Parameters: {known_info['params']:,}")
    print(f"  Best PSNR (training): {known_info['best_psnr']}")

    learned_model, learned_info = load_model(
        args.learned_ckpt, args.n_feat, args.depth,
        known_gradient=False, device=device
    )
    print(f"\nLearned Gradient Model:")
    print(f"  Parameters: {learned_info['params']:,}")
    print(f"  Best PSNR (training): {learned_info['best_psnr']}")

    param_diff = learned_info['params'] - known_info['params']
    print(f"\nParameter difference: {param_diff:,} ({param_diff/known_info['params']*100:.1f}% more for learned)")

    # Load test data
    print(f"\nLoading test data from {args.test_dir}...")
    test_dataset = GaussianDenoiseTestDataset(
        args.test_dir, sigma=args.sigma, center_crop=256
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test images: {len(test_dataset)}")

    # Evaluate
    print("\nEvaluating models...")
    known_results = evaluate_model(known_model, test_loader, device, args.amp)
    learned_results = evaluate_model(learned_model, test_loader, device, args.amp)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'Params':<15}")
    print("-"*70)
    print(f"{'Known Gradient':<20} {known_results['psnr']:.2f} ± {known_results['psnr_std']:.2f}   "
          f"{known_results['ssim']:.4f} ± {known_results['ssim_std']:.4f}   {known_info['params']:,}")
    print(f"{'Learned Gradient':<20} {learned_results['psnr']:.2f} ± {learned_results['psnr_std']:.2f}   "
          f"{learned_results['ssim']:.4f} ± {learned_results['ssim_std']:.4f}   {learned_info['params']:,}")
    print("-"*70)

    psnr_diff = learned_results['psnr'] - known_results['psnr']
    print(f"\nDifference (Learned - Known): {psnr_diff:+.2f} dB PSNR")

    if abs(psnr_diff) < 0.1:
        print("Conclusion: No significant difference - known gradient is preferred (fewer params)")
    elif psnr_diff > 0:
        print(f"Conclusion: Learned gradient is better by {psnr_diff:.2f} dB")
    else:
        print(f"Conclusion: Known gradient is better by {-psnr_diff:.2f} dB")

    # Create visual comparison
    print("\nCreating visual comparison...")
    visualize_comparison(
        known_model, learned_model, test_dataset, device,
        os.path.join(args.save_dir, 'gradient_comparison.png'),
        args.amp
    )

    # Save results to file
    results_file = os.path.join(args.save_dir, 'gradient_ablation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Known vs Learned Gradient Ablation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test data: {args.test_dir}\n")
        f.write(f"Noise level: σ={args.sigma}\n")
        f.write(f"Test images: {len(test_dataset)}\n\n")
        f.write(f"Known Gradient:\n")
        f.write(f"  PSNR: {known_results['psnr']:.2f} ± {known_results['psnr_std']:.2f} dB\n")
        f.write(f"  SSIM: {known_results['ssim']:.4f} ± {known_results['ssim_std']:.4f}\n")
        f.write(f"  Params: {known_info['params']:,}\n\n")
        f.write(f"Learned Gradient:\n")
        f.write(f"  PSNR: {learned_results['psnr']:.2f} ± {learned_results['psnr_std']:.2f} dB\n")
        f.write(f"  SSIM: {learned_results['ssim']:.4f} ± {learned_results['ssim_std']:.4f}\n")
        f.write(f"  Params: {learned_info['params']:,}\n\n")
        f.write(f"Difference: {psnr_diff:+.2f} dB PSNR\n")
    print(f"Saved results to {results_file}")


if __name__ == '__main__':
    main()
