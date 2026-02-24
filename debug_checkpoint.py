"""
Debug script to diagnose SIDD evaluation discrepancy.
Run this to check what's in your checkpoint and verify data loading.
"""

import os
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

from DGUNet import DGUNet
from dataset_denoise import SIDDTestDataset
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inspect_checkpoint(ckpt_path):
    """Inspect checkpoint metadata."""
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {ckpt_path}")
    print('='*60)

    if not os.path.exists(ckpt_path):
        print("ERROR: Checkpoint not found!")
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print(f"Keys in checkpoint: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best PSNR: {ckpt.get('best_psnr', 'N/A')}")
    print(f"Best SSIM: {ckpt.get('best_ssim', 'N/A')}")
    print(f"Global step: {ckpt.get('global_step', 'N/A')}")

    # Check state dict keys to infer architecture
    state_dict = ckpt.get('state_dict', ckpt)
    first_key = list(state_dict.keys())[0]
    print(f"\nFirst state_dict key: {first_key}")

    # Try to infer n_feat from state dict
    for key in state_dict.keys():
        if 'shallow_feat1.0.weight' in key:
            shape = state_dict[key].shape
            print(f"Inferred n_feat from {key}: {shape[0]}")
            break

    return ckpt


def check_sidd_data(sidd_dir, split='val'):
    """Check SIDD data loading."""
    print(f"\n{'='*60}")
    print(f"SIDD DATASET CHECK (split={split})")
    print('='*60)

    data_dir = os.path.join(sidd_dir, 'Data')
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}")
        print(f"Expected structure: {sidd_dir}/Data/0001_xxx/.../GT_SRGB_xxx.PNG")
        return None

    # List scenes
    all_scenes = sorted([d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Total scenes found: {len(all_scenes)}")

    if split == 'val':
        scenes = all_scenes[140:]
    elif split == 'train':
        scenes = all_scenes[:140]
    else:
        scenes = all_scenes

    print(f"Using {len(scenes)} scenes for split='{split}'")

    # Count image pairs
    n_pairs = 0
    for scene in scenes[:3]:  # Check first 3
        scene_path = os.path.join(data_dir, scene)
        gt_files = [f for f in os.listdir(scene_path) if f.startswith('GT_SRGB')]
        print(f"  {scene}: {len(gt_files)} GT images")
        n_pairs += len(gt_files)

    # Load one sample
    dataset = SIDDTestDataset(sidd_dir, center_crop=256, split=split)
    print(f"\nDataset length: {len(dataset)} pairs")

    clean, noisy, name = dataset[0]
    print(f"Sample image shape: {clean.shape}")
    print(f"Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
    print(f"Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}]")

    return dataset


def evaluate_with_debug(model, sidd_dir, split='val', use_amp=True):
    """Evaluate with detailed debugging."""
    print(f"\n{'='*60}")
    print(f"EVALUATION (split={split}, AMP={use_amp})")
    print('='*60)

    dataset = SIDDTestDataset(sidd_dir, center_crop=256, split=split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    psnrs_torch = []
    psnrs_skimage = []

    with torch.no_grad():
        for i, (clean, noisy, name) in enumerate(tqdm(loader, desc='Evaluating')):
            clean = clean.to(device)
            noisy = noisy.to(device)

            # Pad to multiple of 16
            _, _, h, w = noisy.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                noisy = torch.nn.functional.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')

            # Forward pass
            if use_amp:
                with autocast(enabled=True):
                    restored = model(noisy)[0]
            else:
                restored = model(noisy)[0]

            # Crop back
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1)

            # PSNR using torch method (same as training)
            psnr_torch = utils.torchPSNR(restored[0], clean[0]).item()
            psnrs_torch.append(psnr_torch)

            # PSNR using skimage (same as notebook)
            res_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            cln_np = clean[0].cpu().numpy().transpose(1, 2, 0)
            psnr_skimage = compare_psnr(cln_np, res_np, data_range=1.0)
            psnrs_skimage.append(psnr_skimage)

            if i < 3:
                print(f"  Image {i} ({name[0]}): torch={psnr_torch:.2f}, skimage={psnr_skimage:.2f}")

    print(f"\nResults (AMP={use_amp}):")
    print(f"  utils.torchPSNR:  {np.mean(psnrs_torch):.2f} dB")
    print(f"  skimage compare_psnr: {np.mean(psnrs_skimage):.2f} dB")

    return np.mean(psnrs_torch), np.mean(psnrs_skimage)


def main():
    # Configuration - UPDATE THESE PATHS
    SIDD_DIR = './Datasets/SIDD_Small_sRGB_Only'
    CKPT_PATH = './checkpoints/DGUNet-SIDD-DIV2K-7-stages_sigma25/model_best.pth'

    print("="*60)
    print("SIDD EVALUATION DEBUGGING SCRIPT")
    print("="*60)

    # Step 1: Inspect checkpoint
    ckpt = inspect_checkpoint(CKPT_PATH)
    if ckpt is None:
        return

    # Step 2: Check SIDD data
    dataset = check_sidd_data(SIDD_DIR, split='val')
    if dataset is None:
        return

    # Also check what training used (all scenes)
    print("\n--- Checking ALL scenes (what training might have used) ---")
    check_sidd_data(SIDD_DIR, split=None)

    # Step 3: Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print('='*60)

    model = DGUNet(n_feat=80, scale_unetfeats=48, depth=5).to(device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v

    try:
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully!")
    except RuntimeError as e:
        print(f"ERROR loading model: {e}")
        print("\nThis suggests n_feat mismatch! Try different n_feat values.")
        return

    # Step 4: Evaluate with different settings
    print("\n" + "="*60)
    print("COMPARING EVALUATION SETTINGS")
    print("="*60)

    # Test 1: Val split with AMP (should match training)
    psnr_amp_val, _ = evaluate_with_debug(model, SIDD_DIR, split='val', use_amp=True)

    # Test 2: Val split without AMP (current notebook behavior)
    psnr_noamp_val, _ = evaluate_with_debug(model, SIDD_DIR, split='val', use_amp=False)

    # Test 3: ALL scenes with AMP (if training didn't use --sidd_split)
    psnr_amp_all, _ = evaluate_with_debug(model, SIDD_DIR, split=None, use_amp=True)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Checkpoint best_psnr: {ckpt.get('best_psnr', 'N/A')}")
    print(f"Val split + AMP:      {psnr_amp_val:.2f} dB")
    print(f"Val split + no AMP:   {psnr_noamp_val:.2f} dB")
    print(f"All scenes + AMP:     {psnr_amp_all:.2f} dB")
    print()

    if abs(psnr_amp_all - ckpt.get('best_psnr', 0)) < 1.0:
        print("DIAGNOSIS: Training validated on ALL scenes, not val split!")
        print("FIX: Either re-train with --sidd_split, or evaluate on all scenes")
    elif abs(psnr_amp_val - ckpt.get('best_psnr', 0)) < 1.0:
        print("DIAGNOSIS: Results match! The notebook needs AMP enabled.")
    else:
        print("DIAGNOSIS: Unknown issue. Check model architecture parameters.")


if __name__ == '__main__':
    main()
