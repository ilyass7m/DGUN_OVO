"""
Dataset classes for Gaussian denoising experiments.

Supports two modes:
  1. Synthetic noise: loads clean images and adds Gaussian noise on-the-fly
  2. Paired data: loads pre-existing (noisy, clean) pairs (SIDD-style)

Recommended datasets:
  - Training: DIV2K train (800 images, 2K resolution) or BSD400
  - Validation: DIV2K valid (100 images) - NEW DATA not in original paper
  - Testing: BSD68, Kodak24, Urban100 (paper benchmarks)

DIV2K download: https://data.vision.ee.ethz.ch/cvl/DIV2K/
  - DIV2K_train_HR.zip -> extract to Datasets/DIV2K/train/
  - DIV2K_valid_HR.zip -> extract to Datasets/DIV2K/valid/
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpeg', '.JPEG', '.jpg', '.png', '.JPG', '.PNG', '.bmp', '.tif'])


class GaussianDenoiseTrainDataset(Dataset):
    """
    Loads clean images from a directory and adds Gaussian noise on-the-fly.
    Used for training with BSD400, DIV2K, etc.

    Args:
        clean_dir: Path to directory with clean images
        patch_size: Size of random crops (default 128)
        sigma: Noise level (default 25)
        augment: Enable data augmentation (flips + rotations)
        patches_per_image: Number of patches to extract per image per epoch.
                          For DIV2K (800 images), use 8-16 to get more patches per epoch.
                          Total samples per epoch = num_images * patches_per_image
    """
    def __init__(self, clean_dir, patch_size=128, sigma=25, augment=True, patches_per_image=1):
        super().__init__()
        self.clean_paths = sorted([
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if is_image_file(f)
        ])
        self.patch_size = patch_size
        self.sigma = sigma  # noise level (will be divided by 255)
        self.augment = augment
        self.patches_per_image = patches_per_image

    def __len__(self):
        # Return total number of patches (not images)
        return len(self.clean_paths) * self.patches_per_image

    def __getitem__(self, idx):
        # Map idx to image index (allows multiple patches per image)
        img_idx = idx % len(self.clean_paths)
        clean_img = Image.open(self.clean_paths[img_idx]).convert('RGB')
        clean = TF.to_tensor(clean_img)  # [0, 1]

        # Pad if image is smaller than patch_size
        _, h, w = clean.shape
        ps = self.patch_size
        padh = max(ps - h, 0)
        padw = max(ps - w, 0)
        if padh > 0 or padw > 0:
            clean = TF.pad(clean, (0, 0, padw, padh), padding_mode='reflect')

        # Random crop
        _, h, w = clean.shape
        rr = random.randint(0, h - ps)
        cc = random.randint(0, w - ps)
        clean = clean[:, rr:rr+ps, cc:cc+ps]

        # Data augmentation (flips + rotations)
        if self.augment:
            aug = random.randint(0, 7)
            if aug == 1:
                clean = clean.flip(1)
            elif aug == 2:
                clean = clean.flip(2)
            elif aug == 3:
                clean = torch.rot90(clean, dims=(1, 2))
            elif aug == 4:
                clean = torch.rot90(clean, dims=(1, 2), k=2)
            elif aug == 5:
                clean = torch.rot90(clean, dims=(1, 2), k=3)
            elif aug == 6:
                clean = torch.rot90(clean.flip(1), dims=(1, 2))
            elif aug == 7:
                clean = torch.rot90(clean.flip(2), dims=(1, 2))

        # Add Gaussian noise
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise

        return clean, noisy


class GaussianDenoiseTestDataset(Dataset):
    """
    Loads clean test images (BSD68, Set12, Urban100, Kodak24, etc.)
    and adds Gaussian noise with a fixed seed for reproducibility.

    Args:
        clean_dir: Path to directory with clean images
        sigma: Noise level (default 25)
        max_size: If set, resize images so max dimension <= max_size (saves memory for 2K images)
        center_crop: If set, take a center crop of this size instead of full image
    """
    def __init__(self, clean_dir, sigma=25, max_size=None, center_crop=None):
        super().__init__()
        self.clean_paths = sorted([
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if is_image_file(f)
        ])
        self.sigma = sigma
        self.max_size = max_size
        self.center_crop = center_crop

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')

        # Resize large images to save memory (for DIV2K 2K images)
        if self.max_size is not None:
            w, h = clean_img.size
            if max(w, h) > self.max_size:
                scale = self.max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                clean_img = clean_img.resize((new_w, new_h), Image.BICUBIC)

        # Center crop for validation (faster than full image)
        if self.center_crop is not None:
            clean_img = TF.center_crop(clean_img, (self.center_crop, self.center_crop))

        clean = TF.to_tensor(clean_img)

        # Reproducible noise per image
        rng = torch.Generator()
        rng.manual_seed(idx)
        noise = torch.randn(clean.shape, generator=rng) * (self.sigma / 255.0)
        noisy = clean + noise

        filename = os.path.splitext(os.path.basename(self.clean_paths[idx]))[0]
        return clean, noisy, filename


class PairedDenoiseDataset(Dataset):
    """
    Loads paired (noisy, clean) images from input/ and target/ subdirectories.
    Compatible with SIDD-style directory layout.

    Directory structure:
        rgb_dir/
            input/   <- noisy images
            target/  <- clean images
    """
    def __init__(self, rgb_dir, patch_size=128, augment=True):
        super().__init__()
        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        self.inp_paths = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_paths = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.tar_paths)

    def __getitem__(self, idx):
        inp_img = TF.to_tensor(Image.open(self.inp_paths[idx]).convert('RGB'))
        tar_img = TF.to_tensor(Image.open(self.tar_paths[idx]).convert('RGB'))

        ps = self.patch_size
        _, h, w = tar_img.shape
        padh = max(ps - h, 0)
        padw = max(ps - w, 0)
        if padh > 0 or padw > 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        _, h, w = tar_img.shape
        rr = random.randint(0, h - ps)
        cc = random.randint(0, w - ps)
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        if self.augment:
            aug = random.randint(0, 7)
            if aug == 1:
                inp_img, tar_img = inp_img.flip(1), tar_img.flip(1)
            elif aug == 2:
                inp_img, tar_img = inp_img.flip(2), tar_img.flip(2)
            elif aug == 3:
                inp_img = torch.rot90(inp_img, dims=(1, 2))
                tar_img = torch.rot90(tar_img, dims=(1, 2))
            elif aug == 4:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            elif aug == 5:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            elif aug == 6:
                inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            elif aug == 7:
                inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        # Return (clean, noisy) to match GaussianDenoiseTrainDataset interface
        return tar_img, inp_img


class PairedDenoiseTestDataset(Dataset):
    """
    Loads paired (noisy, clean) test images from input/ and target/ subdirectories.
    For SIDD validation set evaluation.
    """
    def __init__(self, rgb_dir, center_crop=None):
        super().__init__()
        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        self.inp_paths = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_paths = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.center_crop = center_crop

    def __len__(self):
        return len(self.tar_paths)

    def __getitem__(self, idx):
        inp_img = Image.open(self.inp_paths[idx]).convert('RGB')
        tar_img = Image.open(self.tar_paths[idx]).convert('RGB')

        if self.center_crop is not None:
            inp_img = TF.center_crop(inp_img, (self.center_crop, self.center_crop))
            tar_img = TF.center_crop(tar_img, (self.center_crop, self.center_crop))

        noisy = TF.to_tensor(inp_img)
        clean = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.basename(self.tar_paths[idx]))[0]
        return clean, noisy, filename


# =============================================================================
# SIDD Dataset Classes (native structure support)
# =============================================================================

class SIDDTrainDataset(Dataset):
    """
    Loads SIDD dataset in its native structure (no preprocessing needed).

    SIDD structure:
        sidd_dir/Data/
            0001_001_S6_.../
                GT_SRGB_010.PNG      <- clean
                NOISY_SRGB_010.PNG   <- noisy
            0002_001_S6_.../
                ...

    Args:
        sidd_dir: Path to SIDD root (containing 'Data' folder)
        patch_size: Size of random crops (default 128)
        augment: Enable data augmentation
        split: 'train', 'val', or None (use all). Train uses first 140, val uses last 20.
    """
    def __init__(self, sidd_dir, patch_size=128, augment=True, split=None):
        super().__init__()
        data_dir = os.path.join(sidd_dir, 'Data')

        # Get all scene folders
        all_scenes = sorted([d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))])

        # Split: 140 train, 20 val (out of 160 total)
        if split == 'train':
            scenes = all_scenes[:140]
        elif split == 'val':
            scenes = all_scenes[140:]
        else:
            scenes = all_scenes

        self.noisy_paths = []
        self.clean_paths = []

        # Scan selected scene folders
        for scene_folder in scenes:
            scene_path = os.path.join(data_dir, scene_folder)

            # Find GT and NOISY images in this scene
            for f in os.listdir(scene_path):
                if f.startswith('GT_SRGB') and f.endswith('.PNG'):
                    gt_path = os.path.join(scene_path, f)
                    # Corresponding noisy image
                    noisy_name = f.replace('GT_SRGB', 'NOISY_SRGB')
                    noisy_path = os.path.join(scene_path, noisy_name)
                    if os.path.exists(noisy_path):
                        self.clean_paths.append(gt_path)
                        self.noisy_paths.append(noisy_path)

        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = TF.to_tensor(Image.open(self.clean_paths[idx]).convert('RGB'))
        noisy_img = TF.to_tensor(Image.open(self.noisy_paths[idx]).convert('RGB'))

        ps = self.patch_size
        _, h, w = clean_img.shape

        # Pad if needed
        padh = max(ps - h, 0)
        padw = max(ps - w, 0)
        if padh > 0 or padw > 0:
            clean_img = TF.pad(clean_img, (0, 0, padw, padh), padding_mode='reflect')
            noisy_img = TF.pad(noisy_img, (0, 0, padw, padh), padding_mode='reflect')

        # Random crop
        _, h, w = clean_img.shape
        rr = random.randint(0, h - ps)
        cc = random.randint(0, w - ps)
        clean_img = clean_img[:, rr:rr+ps, cc:cc+ps]
        noisy_img = noisy_img[:, rr:rr+ps, cc:cc+ps]

        # Data augmentation
        if self.augment:
            aug = random.randint(0, 7)
            if aug == 1:
                clean_img, noisy_img = clean_img.flip(1), noisy_img.flip(1)
            elif aug == 2:
                clean_img, noisy_img = clean_img.flip(2), noisy_img.flip(2)
            elif aug == 3:
                clean_img = torch.rot90(clean_img, dims=(1, 2))
                noisy_img = torch.rot90(noisy_img, dims=(1, 2))
            elif aug == 4:
                clean_img = torch.rot90(clean_img, dims=(1, 2), k=2)
                noisy_img = torch.rot90(noisy_img, dims=(1, 2), k=2)
            elif aug == 5:
                clean_img = torch.rot90(clean_img, dims=(1, 2), k=3)
                noisy_img = torch.rot90(noisy_img, dims=(1, 2), k=3)
            elif aug == 6:
                clean_img = torch.rot90(clean_img.flip(1), dims=(1, 2))
                noisy_img = torch.rot90(noisy_img.flip(1), dims=(1, 2))
            elif aug == 7:
                clean_img = torch.rot90(clean_img.flip(2), dims=(1, 2))
                noisy_img = torch.rot90(noisy_img.flip(2), dims=(1, 2))

        return clean_img, noisy_img


class SIDDTestDataset(Dataset):
    """
    Loads SIDD test/validation set in native structure.

    Args:
        sidd_dir: Path to SIDD root (containing 'Data' folder)
        center_crop: Optional center crop size for memory efficiency
        split: 'train', 'val', or None (use all). Train uses first 140, val uses last 20.
    """
    def __init__(self, sidd_dir, center_crop=None, split=None):
        super().__init__()
        data_dir = os.path.join(sidd_dir, 'Data')

        # Get all scene folders
        all_scenes = sorted([d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))])

        # Split: 140 train, 20 val (out of 160 total)
        if split == 'train':
            scenes = all_scenes[:140]
        elif split == 'val':
            scenes = all_scenes[140:]
        else:
            scenes = all_scenes

        self.noisy_paths = []
        self.clean_paths = []
        self.scene_names = []

        for scene_folder in scenes:
            scene_path = os.path.join(data_dir, scene_folder)

            for f in os.listdir(scene_path):
                if f.startswith('GT_SRGB') and f.endswith('.PNG'):
                    gt_path = os.path.join(scene_path, f)
                    noisy_name = f.replace('GT_SRGB', 'NOISY_SRGB')
                    noisy_path = os.path.join(scene_path, noisy_name)
                    if os.path.exists(noisy_path):
                        self.clean_paths.append(gt_path)
                        self.noisy_paths.append(noisy_path)
                        self.scene_names.append(scene_folder)

        self.center_crop = center_crop

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        noisy_img = Image.open(self.noisy_paths[idx]).convert('RGB')

        if self.center_crop is not None:
            clean_img = TF.center_crop(clean_img, (self.center_crop, self.center_crop))
            noisy_img = TF.center_crop(noisy_img, (self.center_crop, self.center_crop))

        clean = TF.to_tensor(clean_img)
        noisy = TF.to_tensor(noisy_img)

        return clean, noisy, self.scene_names[idx]
