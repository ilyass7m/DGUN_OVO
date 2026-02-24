# Deep Generalized Unfolding Networks for Image Denoising

**OVO Final Project - CentraleSupélec**

Reproduction and ablation study of [DGUNet (CVPR 2022)](https://arxiv.org/abs/2204.13348) for Gaussian color image denoising.

[![WandB Report](https://img.shields.io/badge/WandB-Report-yellow)](https://api.wandb.ai/links/ilyass7m-centralesup-lec/8quoe741)

---

## Overview 

This repository implements **Deep Generalized Unfolding Networks (DGUNet)** for image denoising, bridging classical optimization algorithms with deep learning through the **deep unfolding** paradigm.

### Key Contributions

1. **Complete DGUNet implementation** for Gaussian denoising with support for both synthetic and real noise (SIDD)
2. **Known vs Learned Gradient ablation** - exploiting the analytical gradient for denoising-specific efficiency
3. **Inter-Stage Feature Fusion (ISFF) ablation** - quantifying the contribution of cross-stage information flow
4. **Mixed-precision training (AMP)** with numerical stability fixes for MergeBlock matrix inversion
5. **Comprehensive evaluation tools** including stage-by-stage visualization and cross-domain analysis

---

## Method

### Mathematical Foundation

Image denoising is formulated as solving the variational optimization problem:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{x} - \mathbf{y}\|_2^2 + \lambda \Phi(\mathbf{x})$$

where $\mathbf{y} = \mathbf{x} + \mathbf{n}$ is the noisy observation and $\Phi(\mathbf{x})$ encodes prior knowledge.

### Proximal Gradient Descent (PGD)

PGD solves this by alternating gradient and proximal steps:

```
z^(k) = x^(k-1) - ρ ∇f(x^(k-1))     (Gradient step)
x^(k) = prox_{λΦ}(z^(k))            (Proximal step)
```

For denoising: `∇f(x) = x - y` (analytically known).

### Deep Unfolding: Algorithm → Neural Network

DGUNet unfolds K PGD iterations into K network stages:

| PGD Component | DGUNet Module | Description |
|---------------|---------------|-------------|
| Gradient ∇f | **GDM** (Gradient Descent Module) | Known or learned gradient computation |
| Step size ρ | **r_k** | Learnable per-stage parameter |
| Proximal operator | **PMM** (Proximal Mapping Module) | U-Net encoder-decoder |
| — | **ISFF** (Inter-Stage Feature Fusion) | MergeBlock + CSFF for stage connectivity |

![Architecture](figures/network.PNG)
*DGUNet architecture with 7 unfolding stages. Each stage corresponds to one PGD iteration.*

---

## Installation

```bash
# Clone repository
git clone https://github.com/ilyass7m/DGUN_OVO.git
cd Denoising_OVO

# Create environment
conda create -n dgun python=3.10
conda activate dgun

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image matplotlib tqdm wandb yacs pillow
```

---

## Datasets

### Download Links

| Dataset | Description | Download |
|---------|-------------|----------|
| **DIV2K** | 800 train + 100 val high-resolution images | [Official](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| **SIDD Small** | 160 real noisy/clean smartphone image pairs | [Official](https://www.eecs.yorku.ca/~kalMDCNN/datasets/SIDD_Small_sRGB_Only.zip) |

### Dataset Structure

After downloading, organize datasets as follows:

```
Denoising_OVO/
└── Datasets/
    ├── DIV2K_train_HR/          # 800 training images
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── DIV2K_valid_HR/          # 100 validation images
    │   ├── 0801.png
    │   ├── 0802.png
    │   └── ...
    ├── SIDD_Small_sRGB_Only/    # Real noise dataset
    │   ├── Data/
    │   │   ├── 0001_001_S6_.../
    │   │   │   ├── GT_SRGB_010.PNG      # Ground truth
    │   │   │   └── NOISY_SRGB_010.PNG   # Noisy image
    │   │   ├── 0002_001_S6_.../
    │   │   └── ... (160 scenes)
    │   └── Scene_Instances.txt
    └── own_images/              # Your test images (optional)
        ├── photo1.jpg
        └── ...
```

### Quick Setup

```bash
# Create dataset directories
mkdir -p Datasets/DIV2K_train_HR Datasets/DIV2K_valid_HR Datasets/SIDD_Small_sRGB_Only Datasets/own_images

# Download DIV2K (example with wget)
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_train_HR.zip -d Datasets/
unzip DIV2K_valid_HR.zip -d Datasets/

# Download SIDD Small
wget https://www.eecs.yorku.ca/~kaIDCNN/datasets/SIDD_Small_sRGB_Only.zip
unzip SIDD_Small_sRGB_Only.zip -d Datasets/
```

---

## Pretrained Checkpoints

Download pretrained models from Google Drive:

| Model | Training Data | PSNR | Download |
|-------|---------------|------|----------|
| **DGUNet (σ=25)** | DIV2K synthetic | 33.99 dB | [Google Drive](https://drive.google.com/file/d/1iamIhlBdwY9nvh3A-wI1ANwFZRCKYlCx/view?usp=drive_link) |
| **DGUNet (SIDD)** | SIDD real noise | 37.88 dB | [Google Drive](https://drive.google.com/file/d/1drCuUrozKryvas3ByZE3bzsYmTOC1Oj7/view?usp=drive_link) |
| **DGUNet (Known Grad)** | DIV2K σ=15 | 34.16 dB | [Google Drive](https://drive.google.com/file/d/1kBJ04_H5ENOFAGgWNz_-LmYAXh9Eza5t/view?usp=drive_link) |

### Checkpoint Structure

Place downloaded checkpoints as follows:

```
Denoising_OVO/
└── checkpoints/
    ├── DGUNet-DIV2K-7-stages_sigma25/
    │   └── model_best.pth
    ├── DGUNet-SIDD-DIV2K-7-stages_sigma25/
    │   └── model_best.pth
    └── ablation_DGUN_known_A_DIV2K_sigma15/
        └── model_best.pth
```





## Training

### Quick Start (Synthetic Gaussian Noise)

```bash
python train.py \
    --dataset_mode synthetic \
    --train_dir ./Datasets/DIV2K_train_HR \
    --val_dir ./Datasets/DIV2K_valid_HR \
    --sigma 25 \
    --amp \
    --wandb \
    --name dgunet_sigma25
```

### Training on SIDD (Real Noise)

```bash
python train.py \
    --dataset_mode sidd \
    --train_dir ./Datasets/SIDD_Small_sRGB_Only \
    --val_dir ./Datasets/SIDD_Small_sRGB_Only \
    --sidd_split \
    --amp \
    --wandb \
    --name dgunet_sidd
```

### Full Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_mode` | `synthetic` | `synthetic`, `paired`, or `sidd` |
| `--sigma` | 25 | Noise level for synthetic mode |
| `--batch_size` | 4 | Batch size |
| `--patch_size` | 128 | Training patch size |
| `--epochs` | 80 | Number of epochs |
| `--lr` | 2e-4 | Initial learning rate |
| `--n_feat` | 80 | Feature channels (32, 64, 80) |
| `--depth` | 5 | Unfolding depth (gives 7 stages) |
| `--amp` | False | Enable mixed-precision training |
| `--wandb` | False | Enable Weights & Biases logging |

### Performance Options

```bash
# AMP (2-3x speedup, ~40% less VRAM)
python train.py ... --amp

# torch.compile (PyTorch 2.0+, 10-30% speedup)
python train.py ... --compile

# Combined
python train.py ... --amp --compile
```

---

## Ablation Studies

### 1. Known vs Learned Gradient

For denoising where H=I, the gradient is analytically known: `∇f(x) = x - y`

```bash
# Known gradient (fewer parameters, exact gradient)
python train.py --known_gradient --name ablation_known_grad ...

# Learned gradient (original DGUNet, generalizable)
python train.py --name ablation_learned_grad ...
```

**Results (σ=15, DIV2K):**

| Gradient | PSNR (dB) | SSIM | Parameters |
|----------|-----------|------|------------|
| Known | 34.16 | 0.932 | 17.31M |
| Learned | 33.99 | 0.929 | 17.33M |

### 2. Inter-Stage Feature Fusion (ISFF)

```bash
# Without ISFF
python train.py --no_isff --name ablation_no_isff ...

# With ISFF (default)
python train.py --name ablation_with_isff ...
```

**Results (σ=25, DIV2K):**

| Configuration | PSNR (dB) | SSIM |
|---------------|-----------|------|
| With ISFF | 31.06 | 0.877 |
| Without ISFF | 30.52 | 0.857 |
| **ISFF gain** | **+0.55** | **+0.020** |

### 3. Feature Channels (n_feat)

```bash
python train.py --n_feat 32 --name ablation_nfeat32 ...
python train.py --n_feat 64 --name ablation_nfeat64 ...
python train.py --n_feat 80 --name ablation_nfeat80 ...
```

---

## Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
    --checkpoint ./checkpoints/DGUNet-DIV2K-7-stages_sigma25/model_best.pth \
    --dataset_dir ./Datasets/DIV2K_valid_HR \
    --sigma 25
```

### Visualize Stage-by-Stage Reconstruction

```bash
python visualize_stages.py \
    --checkpoint ./checkpoints/DGUNet-DIV2K-7-stages_sigma25/model_best.pth \
    --image ./Datasets/DIV2K_valid_HR/0801.png \
    --sigma 25
```

### Test on Your Own Images

```bash
python test_own_images.py \
    --checkpoint ./checkpoints/DGUNet-DIV2K-7-stages_sigma25/model_best.pth \
    --image_dir ./Datasets/own_images \
    --sigma 25
```

---


---

## Citation

```bibtex
@inproceedings{mou2022dgunet,
  title={Deep Generalized Unfolding Networks for Image Restoration},
  author={Mou, Chong and Wang, Qian and Zhang, Jian},
  booktitle={CVPR},
  year={2022}
}
```

---

## Acknowledgments

- Original DGUNet implementation: [MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration)
- OVO Course, CentraleSupélec

---

## License

This project is for educational purposes as part of the OVO course at CentraleSupélec.
