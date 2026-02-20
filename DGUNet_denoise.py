"""
DGUNet variant with Known Gradient option for Gaussian Denoising.

For denoising, the degradation model is: y = x + n (H = I)
This means the gradient is analytically known: ∇f(x) = x - y

This variant allows switching between:
  - known_gradient=False: Learn φ and φᵀ as ResBlocks (original, general)
  - known_gradient=True: Use analytical gradient x - y (denoising-specific)

The known gradient version:
  - Has fewer parameters (no φ/φᵀ ResBlocks in GDM)
  - Uses exact gradient (no approximation error)
  - Is theoretically optimal for Gaussian denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base modules from original DGUNet
from DGUNet import (
    conv, conv_down, default_conv, ResBlock, CAB, CALayer,
    SAM, mergeblock, Encoder, Decoder, UNetConvBlock, UNetUpBlock
)


##########################################################################
## Basic Block with Known Gradient Option
class Basic_block_denoise(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48,
                 scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False, known_gradient=False):
        super(Basic_block_denoise, self).__init__()
        act = nn.PReLU()
        self.known_gradient = known_gradient

        # Only create φ/φᵀ if using learned gradient
        if not known_gradient:
            self.phi_1 = ResBlock(default_conv, 3, 3)
            self.phit_1 = ResBlock(default_conv, 3, 3)

        self.shallow_feat2 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, 3, True)

    def forward(self, img, stage1_img, feat1, res1, x2_samfeats):
        ## GDM - Gradient Descent Module
        if self.known_gradient:
            # Known gradient for denoising: ∇f(x) = x - y
            # Update: x_new = x - r * (x - y) = x - r*x + r*y = (1-r)*x + r*y
            gradient = stage1_img - img  # x - y
            x2_img = stage1_img - self.r1 * gradient
        else:
            # Learned gradient: φᵀ(φ(x) - y)
            phixsy_2 = self.phi_1(stage1_img) - img
            x2_img = stage1_img - self.r1 * self.phit_1(phixsy_2)

        ## PMM - Proximal Mapping Module
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2, feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
        return x3_samfeats, stage2_img, feat2, res2


##########################################################################
## DGUNet with Known Gradient Option
class DGUNet_Denoise(nn.Module):
    """
    DGUNet variant optimized for Gaussian denoising.

    Args:
        known_gradient: If True, use analytical gradient (x - y) instead of learned φ/φᵀ.
                       This is theoretically optimal for Gaussian denoising where H = I.
        Other args: Same as original DGUNet.
    """
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48,
                 scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False, depth=5, known_gradient=False):
        super(DGUNet_Denoise, self).__init__()

        act = nn.PReLU()
        self.depth = depth
        self.known_gradient = known_gradient

        # Basic block for stages 2-6 (shared)
        self.basic = Basic_block_denoise(
            in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats,
            num_cab, kernel_size, reduction, bias, known_gradient
        )

        # Stage 1 modules
        self.shallow_feat1 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)

        # Stage 7 (final) modules
        self.shallow_feat7 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        )
        self.concat67 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, 3, kernel_size, bias=bias)

        # Only create φ/φᵀ for stages 1 and 7 if using learned gradient
        if not known_gradient:
            self.phi_0 = ResBlock(default_conv, 3, 3)
            self.phit_0 = ResBlock(default_conv, 3, 3)
            self.phi_6 = ResBlock(default_conv, 3, 3)
            self.phit_6 = ResBlock(default_conv, 3, 3)

        # Step sizes (learnable)
        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, img):
        res = []

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        if self.known_gradient:
            # For stage 1, x^0 = y (noisy input), so gradient = y - y = 0
            # This means x1_img = img (first stage just takes input)
            # But we can still apply a small step: x1_img = img - r*(img - img) = img
            # Or more usefully: initialize with input
            x1_img = img
        else:
            phixsy_1 = self.phi_0(img) - img
            x1_img = img - self.r0 * self.phit_0(phixsy_1)

        ## PMM
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)
        res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 2-6 ------------------
        ##-------------------------------------------
        for _ in range(self.depth):
            x2_samfeats, stage1_img, feat1, res1 = self.basic(img, stage1_img, feat1, res1, x2_samfeats)
            res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        if self.known_gradient:
            gradient = stage1_img - img
            x7_img = stage1_img - self.r6 * gradient
        else:
            phixsy_7 = self.phi_6(stage1_img) - img
            x7_img = stage1_img - self.r6 * self.phit_6(phixsy_7)

        ## PMM
        x7 = self.shallow_feat7(x7_img)
        x7_cat = self.concat67(torch.cat([x7, x2_samfeats], 1))
        stage7_img = self.tail(x7_cat) + img
        res.append(stage7_img)

        return res[::-1]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Compare parameter counts
    from DGUNet import DGUNet

    model_original = DGUNet(n_feat=80, depth=5)
    model_known = DGUNet_Denoise(n_feat=80, depth=5, known_gradient=True)
    model_learned = DGUNet_Denoise(n_feat=80, depth=5, known_gradient=False)

    print(f"Original DGUNet:        {count_parameters(model_original):,} parameters")
    print(f"DGUNet (learned grad):  {count_parameters(model_learned):,} parameters")
    print(f"DGUNet (known grad):    {count_parameters(model_known):,} parameters")
    print(f"Parameter reduction:    {count_parameters(model_original) - count_parameters(model_known):,}")

    # Test forward pass
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out_orig = model_original(x)
        out_known = model_known(x)
    print(f"\nOriginal output stages: {len(out_orig)}")
    print(f"Known grad output stages: {len(out_known)}")
    print(f"Output shape: {out_known[0].shape}")
