from torch import nn

from models.layers import Upsampler, ResidualBlock


class EDSR(nn.Module):
    def __init__(self, channels=3, scale_factor=2, blocks=32, base_width=256):
        super(EDSR, self).__init__()

        self.entry = nn.Conv2d(channels, base_width, kernel_size=3, padding=1)

        body = [ResidualBlock(base_width, base_width, activation=None) for _ in range(blocks)]
        body.append(nn.Conv2d(base_width, base_width, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        self.upsample = Upsampler(base_width, base_width, scale_factor)
        self.exit = nn.Conv2d(base_width, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        out = self.body(x)
        out += x
        out = self.upsample(x)
        out = self.exit(out)
        return out
