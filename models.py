"""
models.py
=========

Small U-Net for SSH forecasting. Parameterized so we can create ensemble
members with varied width and depth.

Input:  (B, HISTORY, H, W)   — past N days of SSH
Output: (B, 1, H, W)         — next day's SSH prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs with ReLU, same spatial size."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class SmallUNet(nn.Module):
    """
    Small U-Net. Size controlled by `base_width` and `depth`.

    depth=2, base_width=16:  ~10k params   (tiny)
    depth=3, base_width=32:  ~250k params  (standard)
    depth=4, base_width=32:  ~1M params    (larger)
    """
    def __init__(self, in_channels=7, out_channels=1, base_width=32, depth=3):
        super().__init__()
        self.depth = depth

        # Build encoder
        self.down_blocks = nn.ModuleList()
        widths = [base_width * (2 ** i) for i in range(depth + 1)]
        prev_c = in_channels
        for w in widths[:-1]:
            self.down_blocks.append(ConvBlock(prev_c, w))
            prev_c = w

        # Bottleneck
        self.bottleneck = ConvBlock(widths[-2], widths[-1])

        # Build decoder
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth, 0, -1):
            # Upsample conv (transpose) goes widths[i] -> widths[i-1]
            self.up_convs.append(
                nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2)
            )
            # After concat with skip, channels = 2 * widths[i-1]
            self.up_blocks.append(ConvBlock(2 * widths[i - 1], widths[i - 1]))

        # Final 1x1 to produce output
        self.out_conv = nn.Conv2d(widths[0], out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Pad so H, W are divisible by 2^depth
        _, _, H, W = x.shape
        pad = 2 ** self.depth
        H_pad = (pad - H % pad) % pad
        W_pad = (pad - W % pad) % pad
        if H_pad or W_pad:
            x = F.pad(x, (0, W_pad, 0, H_pad), mode="replicate")

        # Encoder with skip connections
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for up_conv, up_block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up_conv(x)
            # Crop/pad if sizes mismatch (rare with our padding above, but safe)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = up_block(x)

        x = self.out_conv(x)

        # Remove padding
        if H_pad or W_pad:
            x = x[..., :H, :W]
        return x


def build_model(config):
    """Build a model from a config dict."""
    return SmallUNet(
        in_channels=config.get("in_channels", 7),
        out_channels=1,
        base_width=config.get("base_width", 32),
        depth=config.get("depth", 3),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test with your actual field size
    for cfg in [
        {"base_width": 16, "depth": 2},
        {"base_width": 32, "depth": 3},
        {"base_width": 48, "depth": 3},
    ]:
        m = build_model(cfg)
        x = torch.randn(2, 7, 120, 146)
        y = m(x)
        print(f"{cfg}: params={count_params(m):,}, out shape {y.shape}")
    print("models.py OK")
