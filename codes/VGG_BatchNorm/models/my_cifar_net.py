"""
Part 1: Custom CIFAR-10 Network (Strong Version)

Architecture (4 stages, deep channels, ~11M params for L):
  Conv(3→base_ch, 3x3) → BN → ReLU          [32×32]
  Stage1: 2× ResBlock(base_ch → base_ch)      [32×32]
  Stage2: ResBlock(base_ch → base_ch*2, /2)   [16×16]
          1× ResBlock(base_ch*2 → base_ch*2)
  Stage3: ResBlock(base_ch*2 → base_ch*4, /2) [8×8]
          2× ResBlock(base_ch*4 → base_ch*4)
  Stage4: ResBlock(base_ch*4 → base_ch*8, /2) [4×4]
          2× ResBlock(base_ch*8 → base_ch*8)
  AdaptiveAvgPool2d(1) → Flatten
  FC(base_ch*8 → 256) → ReLU → Dropout
  FC(256 → 10)

Variants:
  S: base_ch=32   ~2.8M params
  M: base_ch=48   ~6.3M params
  L: base_ch=64   ~11.2M params

Covering PDF requirements: FC, Conv2d, Pooling, Activation,
BatchNorm, Dropout, Residual Connection.
"""
import torch
from torch import nn


class ResBlock(nn.Module):
    """Conv(3x3)→BN→ReLU→Conv(3x3)→BN → +skip → ReLU"""

    def __init__(self, in_ch, out_ch, stride=1, activation='relu'):
        super().__init__()
        act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

        self.act2 = act_fn

    def forward(self, x):
        identity = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        return out


class MyCIFARNet(nn.Module):
    """Custom strong CNN for CIFAR-10.

    Args:
        base_ch: base channel count (32=S, 48=M, 64=L)
        activation: 'relu' or 'leaky_relu'
        dropout: dropout rate
        stages: list of (n_blocks, mult) for each stage after the first.
                First stage uses base_ch, subsequent use base_ch * mult.
    """

    def __init__(self, base_ch=64, num_classes=10,
                 activation='relu', dropout=0.3):
        super().__init__()
        act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.1)

        # Initial conv
        self.conv1 = nn.Conv2d(3, base_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.act1 = act_fn

        # Define stages: (in_ch, out_ch, n_blocks, stride_first)
        stages_config = [
            (base_ch,      base_ch,     2, 1),   # 32×32
            (base_ch,      base_ch * 2, 2, 2),   # 16×16
            (base_ch * 2,  base_ch * 4, 2, 2),   # 8×8
            (base_ch * 4,  base_ch * 8, 2, 2),   # 4×4
        ]

        self.stages = nn.ModuleList()
        for in_ch, out_ch, n_blocks, stride in stages_config:
            stage = []
            # First block may change channels and spatial size
            stage.append(ResBlock(in_ch, out_ch, stride, activation))
            # Remaining blocks keep dimensions
            for _ in range(n_blocks - 1):
                stage.append(ResBlock(out_ch, out_ch, 1, activation))
            self.stages.append(nn.Sequential(*stage))

        final_ch = base_ch * 8  # 512 for L

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        for stage in self.stages:
            out = stage(out)
        out = self.pool(out)
        return self.classifier(out)


# Predefined variants
def MyCIFARNet_S(**kwargs):
    return MyCIFARNet(base_ch=32, **kwargs)


def MyCIFARNet_M(**kwargs):
    return MyCIFARNet(base_ch=48, **kwargs)


def MyCIFARNet_L(**kwargs):
    return MyCIFARNet(base_ch=64, **kwargs)


if __name__ == '__main__':
    for name, net in [('S', MyCIFARNet_S()), ('M', MyCIFARNet_M()),
                       ('L', MyCIFARNet_L())]:
        params = sum(p.numel() for p in net.parameters()) / 1e6
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        print(f"MyCIFARNet_{name}: {params:.2f}M params, output: {y.shape}")
