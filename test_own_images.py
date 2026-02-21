"""
Testing DGUNet on Your Own Images (Phone Photos)

===========================================
METHODOLOGY FOR TESTING ON PERSONAL IMAGES
===========================================

Since we need ground truth to compute PSNR/SSIM, there are two approaches:

APPROACH 1: Synthetic Noise on Clean Photos (Recommended)
---------------------------------------------------------
1. Take photos with your phone in GOOD lighting conditions
2. Use low ISO settings to minimize real sensor noise
3. The original photo serves as "ground truth"
4. We add controlled Gaussian noise (sigma=15, 25, or 50)
5. DGUNet denoises the artificially noisy image
6. Compare denoised output with original clean photo

    Clean Photo (GT) ---> Add Noise ---> Noisy Image ---> DGUNet ---> Denoised
         |                                                              |
         +----------------------- Compare (PSNR/SSIM) -----------------+

This is valid because:
- DGUNet is trained on synthetic Gaussian noise
- We can precisely control noise level
- We have exact ground truth for evaluation


APPROACH 2: Burst Photography (For Real Noise, Advanced)
--------------------------------------------------------
1. Use burst mode to capture 10-20 photos of a STATIC scene
2. Average all burst photos to create a "pseudo ground truth"
3. Use individual burst frames as noisy inputs
4. This captures real sensor noise patterns

    Burst Frame 1  ---|
    Burst Frame 2  ---|---> Average ---> Pseudo Ground Truth
    ...            ---|
    Burst Frame N  ---|

    Single Burst Frame ---> DGUNet ---> Denoised ---> Compare with Average


Usage:
------
    # Basic usage with your photo
    python test_own_images.py --image_dir ./my_photos/ --sigma 25 --weights checkpoints/model_best.pth

    # Test multiple noise levels
    python test_own_images.py --image_dir ./my_photos/ --sigma 15 25 50 --weights checkpoints/model_best.pth

    # Save denoised outputs
    python test_own_images.py --image ./photo.jpg --sigma 25 --weights checkpoints/model_best.pth --save_outputs
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
from collections import OrderedDict
from skimage.metrics import structural_similarity as compare_ssim
from glob import glob
from datetime import datetime

from DGUNet import DGUNet
from DGUNet_denoise import DGUNet_Denoise


def get_image_metadata(path):
    """Extract EXIF metadata from image (useful for phone photos)."""
    try:
        img = Image.open(path)
        exif_data = img._getexif()
        if exif_data:
            metadata = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ['Make', 'Model', 'ISO', 'ExposureTime', 'FNumber', 'DateTimeOriginal']:
                    metadata[tag] = value
            return metadata
    except:
        pass
    return {}


def load_image(path, max_size=None):
    """Load and preprocess an image."""
    img = Image.open(path).convert('RGB')
    original_size = img.size

    # Resize if too large
    if max_size and max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Ensure dimensions are multiples of 16 (for U-Net)
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w != w or new_h != h:
        # Center crop to valid dimensions
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))

    return np.array(img).astype(np.float32) / 255.0, original_size


def add_gaussian_noise(clean_img, sigma):
    """Add Gaussian noise with specified sigma."""
    np.random.seed(None)  # Different noise each time
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

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model


def denoise_image(model, noisy_img, device):
    """Run denoising and return the final output."""
    noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(noisy_tensor)
        # First output is the final denoised image
        denoised = outputs[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)

    return np.clip(denoised, 0, 1)


def create_comparison_figure(clean, noisy, denoised, sigma, psnr_noisy, psnr_denoised,
                             ssim_noisy, ssim_denoised, img_name, metadata=None):
    """Create a side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Clean (Ground Truth)
    axes[0].imshow(clean)
    axes[0].set_title('Original (Ground Truth)', fontsize=12)
    axes[0].axis('off')

    # Noisy
    axes[1].imshow(noisy)
    axes[1].set_title(f'Noisy (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}', fontsize=12)
    axes[1].axis('off')

    # Denoised
    axes[2].imshow(denoised)
    improvement = psnr_denoised - psnr_noisy
    axes[2].set_title(f'DGUNet Denoised\nPSNR: {psnr_denoised:.2f} dB (+{improvement:.2f}) | SSIM: {ssim_denoised:.4f}', fontsize=12)
    axes[2].axis('off')

    # Add metadata info if available
    meta_str = ""
    if metadata:
        if 'Model' in metadata:
            meta_str += f"Camera: {metadata.get('Make', '')} {metadata['Model']} | "
        if 'ISO' in metadata:
            meta_str += f"ISO: {metadata['ISO']} | "
        if 'ExposureTime' in metadata:
            exp = metadata['ExposureTime']
            if isinstance(exp, tuple):
                meta_str += f"Exposure: {exp[0]}/{exp[1]}s"
            else:
                meta_str += f"Exposure: {exp}s"

    title = f'{img_name}'
    if meta_str:
        title += f'\n{meta_str}'

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def create_zoomed_comparison(clean, noisy, denoised, sigma, crop_coords=None):
    """Create a zoomed comparison to show detail preservation."""
    h, w = clean.shape[:2]

    # Default crop: center region, 1/4 of image
    if crop_coords is None:
        crop_h, crop_w = h // 4, w // 4
        y1, x1 = h // 2 - crop_h // 2, w // 2 - crop_w // 2
        y2, x2 = y1 + crop_h, x1 + crop_w
    else:
        y1, x1, y2, x2 = crop_coords

    # Crop regions
    clean_crop = clean[y1:y2, x1:x2]
    noisy_crop = noisy[y1:y2, x1:x2]
    denoised_crop = denoised[y1:y2, x1:x2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Full images
    axes[0, 0].imshow(clean)
    axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy)
    axes[0, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    axes[0, 1].set_title(f'Noisy (σ={sigma})', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised)
    axes[0, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    axes[0, 2].set_title('DGUNet Denoised', fontsize=12)
    axes[0, 2].axis('off')

    # Zoomed crops
    axes[1, 0].imshow(clean_crop)
    axes[1, 0].set_title('Original (Zoomed)', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_crop)
    axes[1, 1].set_title('Noisy (Zoomed)', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_crop)
    axes[1, 2].set_title('Denoised (Zoomed)', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle(f'Detail Comparison (σ={sigma})', fontsize=14)
    plt.tight_layout()

    return fig


def test_multiple_sigma(model, clean_img, device, sigmas=[15, 25, 50]):
    """Test denoising at multiple noise levels."""
    results = []

    for sigma in sigmas:
        noisy = add_gaussian_noise(clean_img, sigma)
        denoised = denoise_image(model, noisy, device)

        psnr_noisy = compute_psnr(clean_img, noisy)
        psnr_denoised = compute_psnr(clean_img, denoised)
        ssim_noisy = compute_ssim(clean_img, noisy)
        ssim_denoised = compute_ssim(clean_img, denoised)

        results.append({
            'sigma': sigma,
            'noisy': noisy,
            'denoised': denoised,
            'psnr_noisy': psnr_noisy,
            'psnr_denoised': psnr_denoised,
            'ssim_noisy': ssim_noisy,
            'ssim_denoised': ssim_denoised,
            'psnr_gain': psnr_denoised - psnr_noisy
        })

    return results


def print_results_table(results, img_name):
    """Print a formatted results table."""
    print(f"\n{'=' * 70}")
    print(f"Results for: {img_name}")
    print(f"{'=' * 70}")
    print(f"{'Sigma':<10} {'Noisy PSNR':<15} {'Denoised PSNR':<15} {'Gain':<10} {'SSIM':<10}")
    print(f"{'-' * 70}")

    for r in results:
        print(f"{r['sigma']:<10} {r['psnr_noisy']:<15.2f} {r['psnr_denoised']:<15.2f} "
              f"+{r['psnr_gain']:<9.2f} {r['ssim_denoised']:<10.4f}")

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test DGUNet on your own images')

    # Input
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')

    # Noise settings
    parser.add_argument('--sigma', type=int, nargs='+', default=[25],
                        help='Noise level(s) to test (default: 25). Can specify multiple: --sigma 15 25 50')

    # Model
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--n_feat', type=int, default=80, help='Model feature channels')
    parser.add_argument('--depth', type=int, default=5, help='Model depth')
    parser.add_argument('--known_gradient', action='store_true', help='Use known gradient model')

    # Output
    parser.add_argument('--save_dir', type=str, default='./own_images_results', help='Output directory')
    parser.add_argument('--save_outputs', action='store_true', help='Save denoised images')
    parser.add_argument('--max_size', type=int, default=1024, help='Max image dimension')
    parser.add_argument('--show_zoom', action='store_true', help='Show zoomed detail comparison')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Must provide --image or --image_dir")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights, device, args.n_feat, args.depth, args.known_gradient)
    print(f"Model loaded successfully")

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob(os.path.join(args.image_dir, ext)))

    print(f"\nFound {len(image_paths)} image(s) to process")
    print(f"Testing sigma levels: {args.sigma}")

    # Aggregate results
    all_results = {sigma: [] for sigma in args.sigma}

    # Process each image
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing: {img_name}")

        # Get metadata
        metadata = get_image_metadata(img_path)
        if metadata:
            print(f"  Camera: {metadata.get('Model', 'Unknown')}, ISO: {metadata.get('ISO', 'Unknown')}")

        # Load image
        clean_img, original_size = load_image(img_path, args.max_size)
        print(f"  Original size: {original_size}, Processed size: {clean_img.shape[:2][::-1]}")

        # Test at all sigma levels
        results = test_multiple_sigma(model, clean_img, device, args.sigma)
        print_results_table(results, img_name)

        # Store for aggregation
        for r in results:
            all_results[r['sigma']].append(r)

        # Save visualizations
        for r in results:
            sigma = r['sigma']

            # Comparison figure
            fig = create_comparison_figure(
                clean_img, r['noisy'], r['denoised'], sigma,
                r['psnr_noisy'], r['psnr_denoised'],
                r['ssim_noisy'], r['ssim_denoised'],
                img_name, metadata
            )
            fig.savefig(os.path.join(args.save_dir, f'{img_name}_sigma{sigma}_comparison.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Zoomed comparison
            if args.show_zoom:
                fig_zoom = create_zoomed_comparison(clean_img, r['noisy'], r['denoised'], sigma)
                fig_zoom.savefig(os.path.join(args.save_dir, f'{img_name}_sigma{sigma}_zoomed.png'),
                                 dpi=150, bbox_inches='tight')
                plt.close(fig_zoom)

            # Save denoised output
            if args.save_outputs:
                denoised_uint8 = (r['denoised'] * 255).astype(np.uint8)
                Image.fromarray(denoised_uint8).save(
                    os.path.join(args.save_dir, f'{img_name}_sigma{sigma}_denoised.png')
                )

    # Print aggregate statistics
    if len(image_paths) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS (All Images)")
        print("=" * 70)
        print(f"{'Sigma':<10} {'Avg Noisy PSNR':<18} {'Avg Denoised PSNR':<18} {'Avg Gain':<12} {'Avg SSIM':<12}")
        print("-" * 70)

        for sigma in args.sigma:
            results = all_results[sigma]
            avg_noisy_psnr = np.mean([r['psnr_noisy'] for r in results])
            avg_denoised_psnr = np.mean([r['psnr_denoised'] for r in results])
            avg_gain = np.mean([r['psnr_gain'] for r in results])
            avg_ssim = np.mean([r['ssim_denoised'] for r in results])

            print(f"{sigma:<10} {avg_noisy_psnr:<18.2f} {avg_denoised_psnr:<18.2f} "
                  f"+{avg_gain:<11.2f} {avg_ssim:<12.4f}")

        print("=" * 70)

    print(f"\nResults saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
