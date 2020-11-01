import torch
from torch import nn

from models.layers import ResidualBlock, Upsampler, Transition


class CascadeBlock(nn.Module):
    def __init__(self, block, in_channels, out_channels, base_width, n_blocks=3):
        super(CascadeBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(block(in_channels if i == 0 else base_width, base_width))
            self.transitions.append(
                Transition(base_width * (i + 2), out_channels if i == n_blocks - 1 else base_width)
            )

    def forward(self, x):
        c = x
        for block, transition in zip(self.blocks, self.transitions):
            y = block(x)
            c = torch.cat([c, y], dim=1)
            x = transition(c)
        return x


class CARN(nn.Module):
    def __init__(self, channels=3, scale_factor=2, base_width=64, blocks=3, groups=1):
        super(CARN, self).__init__()
        self.base_width = base_width

        self.entry = nn.Conv2d(channels, base_width, kernel_size=3, padding=1)
        self.body = CascadeBlock(self._make_block, base_width, base_width, base_width, n_blocks=blocks)
        self.upsample = Upsampler(base_width, base_width, scale_factor, groups=groups, activation=nn.ReLU(True))
        self.exit = nn.Conv2d(base_width, channels, kernel_size=3, padding=1)

    def _make_block(self, in_channels, out_channels):
        return CascadeBlock(ResidualBlock, in_channels, out_channels, base_width=self.base_width)

    def forward(self, x):
        x = self.entry(x)
        out = self.body(x)
        out = out + x
        out = self.upsample(out)
        out = self.exit(out)

        return out
