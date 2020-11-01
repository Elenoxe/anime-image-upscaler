import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, activation=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.relu = nn.ReLU(True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(x) if self.shortcut else x
        if self.act is not None:
            out = self.act(out)

        return out


class EResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, activation=True):
        super(EResidualBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += self.shortcut(x) if self.shortcut else x
        if self.activation:
            out = self.act(out)

        return out


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor: int, groups=1, activation=None):
        super(Upsampler, self).__init__()
        self.conv = nn.Conv2d(in_channels, (scale_factor ** 2) * out_channels, kernel_size=3, padding=1, groups=groups)
        self.act = activation
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        out = self.conv(x)
        if self.act:
            out = self.act(out)
        out = self.pixel_shuffle(out)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(True)):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = activation

    def forward(self, x):
        out = self.act(self.conv(x))
        return out


class ContrastChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ContrastChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def contrast(self, x: torch.Tensor):
        return x.view(x.shape[0], x.shape[1], -1).std(dim=2, keepdim=True).unsqueeze(2)

    def forward(self, x):
        w = self.contrast(x) + self.avg_pool(x)
        w = self.relu(self.conv1(w))
        w = self.sigmoid(self.conv2(w))
        return w * x
