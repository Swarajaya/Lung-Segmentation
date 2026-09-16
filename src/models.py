"""
U-Net implementation for binary semantic segmentation of lung CT slices.

Architecture (Ronneberger et al., 2015, "U-Net: Convolutional Networks for
Biomedical Image Segmentation"):

    Encoder (contracting path): repeated [Conv-BN-ReLU x2] blocks followed by
    2x2 max-pooling, doubling the channel count at each stage. This builds an
    increasingly abstract, lower-resolution representation of "what" is in
    the image while discarding precise spatial detail.

    Bottleneck: the deepest conv block, operating at the lowest spatial
    resolution and highest channel depth, capturing the most abstract
    features.

    Decoder (expansive path): repeated up-sampling (transposed conv or
    bilinear upsample) followed by concatenation with the corresponding
    encoder feature map (the "skip connection") and another [Conv-BN-ReLU x2]
    block. Skip connections re-inject the precise spatial/boundary
    information that pooling discarded, which is exactly what is needed to
    draw an accurate pixel-level lung/lesion boundary.

    Final layer: a 1x1 convolution mapping to the desired number of output
    channels (1 for binary foreground/background), producing raw logits.

Why U-Net is appropriate here: medical images have limited annotated data,
strong low-level structure (organ/lesion boundaries), and need pixel-precise
output. U-Net's symmetric encoder-decoder with skip connections is exactly
designed for this: it needs relatively few training images (helped further
here by data augmentation), trains end-to-end, and its skip connections
directly address the loss of localization precision that plain
encoder-only or heavily-downsampled architectures suffer from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv (one encoder stage)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then concatenate skip connection then double conv (one decoder stage)."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # pad in case of odd input dimensions
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping to the desired number of output classes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Configurable U-Net.

    base_channels controls model capacity: the classic paper uses 64, we
    default to a smaller value so the model is trainable in constrained
    (CPU-only / smoke-test) environments; raise it via config.yaml for full
    GPU training.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, bilinear=True):
        super().__init__()
        c = base_channels
        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(c * 8, c * 16 // factor)

        self.up1 = Up(c * 16, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)
        self.out_conv = OutConv(c, out_channels)

    def forward(self, x):
        x1 = self.in_conv(x)   # encoder stage 0 (full resolution)
        x2 = self.down1(x1)    # encoder stage 1
        x3 = self.down2(x2)    # encoder stage 2
        x4 = self.down3(x3)    # encoder stage 3
        x5 = self.down4(x4)    # bottleneck

        x = self.up1(x5, x4)   # decoder stage 1 + skip from x4
        x = self.up2(x, x3)    # decoder stage 2 + skip from x3
        x = self.up3(x, x2)    # decoder stage 3 + skip from x2
        x = self.up4(x, x1)    # decoder stage 4 + skip from x1
        logits = self.out_conv(x)
        return logits


# ---------------------------------------------------------------------------
# Optional improved variants.
#
# U-Net above remains the PRIMARY baseline for this project: it is the
# reference architecture for biomedical segmentation, it is simple enough to
# explain and reproduce exactly, and every reported result is anchored to it.
# The two variants below are offered as controlled, single-change ablations,
# not as replacements:
#
#   ResidualUNet  - replaces each DoubleConv with a residual block
#                   (out = ReLU(F(x) + shortcut(x))). Residual connections
#                   ease gradient flow through the deeper encoder, which can
#                   help when training longer on a GPU.
#
#   AttentionUNet - adds an additive attention gate on each skip connection
#                   (Oktay et al., 2018, "Attention U-Net"). The gate uses the
#                   coarse decoder feature as a query to re-weight the fine
#                   encoder feature, suppressing irrelevant background regions
#                   before concatenation. This is motivated directly by this
#                   dataset's extreme class imbalance (mean foreground ~0.45%
#                   of pixels): most of every skip feature map is background.
#
# Both keep the identical input/output contract as UNet, so losses, metrics,
# training loop, evaluation and the Streamlit app work unchanged.
# ---------------------------------------------------------------------------


class ResidualDoubleConv(nn.Module):
    """(Conv-BN-ReLU -> Conv-BN) + projected shortcut, then ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class AttentionGate(nn.Module):
    """
    Additive attention gate.

    g    : gating signal from the coarser decoder level (query)
    skip : encoder feature map to be filtered (value)

    psi = sigma2(psi_conv(ReLU(W_g(g) + W_x(skip)))) in [0,1], broadcast over
    channels, multiplies `skip` element-wise. Regions the decoder does not
    consider relevant are attenuated before concatenation.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, skip):
        if g.shape[-2:] != skip.shape[-2:]:
            g = F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        attn = self.psi(self.relu(self.W_g(g) + self.W_x(skip)))
        return skip * attn


class _UpGeneric(nn.Module):
    """Decoder stage with a pluggable conv block and an optional attention gate."""

    def __init__(self, in_channels, out_channels, bilinear=True,
                 block_cls=DoubleConv, attention=False, skip_channels=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            gate_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            gate_channels = in_channels // 2
        skip_channels = skip_channels if skip_channels is not None else in_channels // 2
        self.attn = (AttentionGate(gate_channels, skip_channels,
                                   max(1, skip_channels // 2)) if attention else None)
        self.conv = block_cls(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        if self.attn is not None:
            skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConfigurableUNet(nn.Module):
    """
    U-Net family with two orthogonal switches:
        residual_blocks : use ResidualDoubleConv instead of DoubleConv
        attention_gates : gate every skip connection

    residual_blocks=False, attention_gates=False reproduces the plain UNet
    topology exactly (same channel widths, same skip structure).
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32,
                 bilinear=True, residual_blocks=False, attention_gates=False):
        super().__init__()
        block = ResidualDoubleConv if residual_blocks else DoubleConv
        c = base_channels
        factor = 2 if bilinear else 1

        self.in_conv = block(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), block(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), block(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), block(c * 4, c * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), block(c * 8, c * 16 // factor))

        self.up1 = _UpGeneric(c * 16, c * 8 // factor, bilinear, block, attention_gates, c * 8)
        self.up2 = _UpGeneric(c * 8, c * 4 // factor, bilinear, block, attention_gates, c * 4)
        self.up3 = _UpGeneric(c * 4, c * 2 // factor, bilinear, block, attention_gates, c * 2)
        self.up4 = _UpGeneric(c * 2, c, bilinear, block, attention_gates, c)
        self.out_conv = OutConv(c, out_channels)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


class ResidualUNet(ConfigurableUNet):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, bilinear=True):
        super().__init__(in_channels, out_channels, base_channels, bilinear,
                         residual_blocks=True, attention_gates=False)


class AttentionUNet(ConfigurableUNet):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, bilinear=True):
        super().__init__(in_channels, out_channels, base_channels, bilinear,
                         residual_blocks=False, attention_gates=True)


MODEL_REGISTRY = {
    "unet": UNet,
    "residual_unet": ResidualUNet,
    "attention_unet": AttentionUNet,
}


def build_model(cfg, name_override: str = None):
    """
    Instantiate the architecture named by cfg.model.name (or `name_override`,
    used by the experiment runner). Unknown names fail loudly rather than
    silently falling back to the baseline, so an experiment can never be
    mislabelled in the results table.
    """
    mc = cfg.model
    name = (name_override or mc.get("name", "unet")).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](
        in_channels=mc.in_channels,
        out_channels=mc.out_channels,
        base_channels=mc.base_channels,
        bilinear=mc.bilinear_upsampling,
    )
