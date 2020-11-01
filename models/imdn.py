import torch
from torch import nn

from models.layers import Upsampler, ContrastChannelAttention


class Distillation(nn.Module):
    def __init__(self, in_channels, rate=0.25):
        super(Distillation, self).__init__()
        self.distilled = int(in_channels * rate)
        self.remaining = in_channels - self.distilled

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.remaining, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.remaining, in_channels, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(self.remaining, self.distilled, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        c1 = self.leaky_relu(self.conv1(x))
        dc1, rc1 = torch.split(c1, (self.distilled, self.remaining), dim=1)

        c2 = self.leaky_relu(self.conv2(rc1))
        dc2, rc2 = torch.split(c2, (self.distilled, self.remaining), dim=1)

        c3 = self.leaky_relu(self.conv3(rc2))
        dc3, rc3 = torch.split(c3, (self.distilled, self.remaining), dim=1)

        c4 = self.conv4(rc3)

        out = torch.cat([dc1, dc2, dc3, c4], dim=1)
        return out


class IMDBlock(nn.Module):
    def __init__(self, channels):
        super(IMDBlock, self).__init__()
        self.distillation = Distillation(channels)
        self.attention = ContrastChannelAttention(self.distillation.distilled * 4)
        self.transition = nn.Conv2d(self.distillation.distilled * 4, channels, kernel_size=1)

    def forward(self, x):
        out = self.distillation(x)
        out = self.attention(out)
        out = self.transition(out)
        return out + x


class IMDN(nn.Module):
    def __init__(self, channels=3, scale_factor=2, base_width=64, blocks=6):
        super(IMDN, self).__init__()
        self.channels = channels
        self.base_width = base_width
        self.entry = nn.Conv2d(channels, base_width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([IMDBlock(base_width) for _ in range(blocks)])
        self.neck = nn.Sequential(
            nn.Conv2d(base_width * blocks, base_width, kernel_size=1),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1)
        )
        self.upsample = Upsampler(base_width, channels, scale_factor=scale_factor)

    def reset_upsampler(self, scale_factor):
        self.upsample = Upsampler(self.base_width, self.channels, scale_factor=scale_factor)

    def forward(self, x):
        x = self.entry(x)
        out = x
        features = []
        for block in self.blocks:
            out = block(out)
            features.append(out)
        out = torch.cat(features, dim=1)
        out = self.neck(out)
        out += x
        out = self.upsample(out)
        return out
