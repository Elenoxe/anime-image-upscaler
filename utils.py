import copy
from io import BytesIO
from typing import Tuple, Union, Iterable

import pytorch_msssim
import torch
from PIL import Image
from numpy import random
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from metrics import WeightedAveragedMetric


class RandomNoise:
    def __init__(self, noise: Union[int, Tuple[int, int]]):
        self.noise = noise if isinstance(noise, Iterable) else (noise, noise)

    def __call__(self, img: Image.Image):
        if self.noise[1] <= 0:
            return img
        quality = 100 - round(random.uniform(*self.noise))
        mem = BytesIO()
        img.save(mem, format='JPEG', quality=quality)
        mem.seek(0)
        out = Image.open(mem)
        return out


class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, seg_size=128, scale_factor=2, boarder_pad_size=3):
        self.seg_size = seg_size
        self.scale_factor = scale_factor
        self.pad_size = boarder_pad_size
        self.height = 0
        self.width = 0

    @torch.no_grad()
    def rescale(self, model: nn.Module, img: Union[torch.Tensor, Image.Image],
                progressbar=False, desc='Rescale') -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_tensor = to_tensor(img)
        else:
            img_tensor = img
        patches = self.split_patches(img_tensor)
        iterated = tqdm(patches, desc=desc) if progressbar else patches
        out = [model(patch) for patch in iterated]
        result = self.merge_patches(out)
        if isinstance(img, Image.Image):
            result = to_pil_image(result)
        return result

    def split_patches(self, img_tensor: torch.Tensor, padding=0):
        # resize image and convert them into tensor
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = nn.ReplicationPad2d(self.pad_size)(img_tensor)
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        patches = []
        # avoid the residual part is smaller than the padded size
        if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
            self.seg_size += self.scale_factor * self.pad_size

        # split image into over-lapping pieces
        for i in range(self.pad_size, height, self.seg_size):
            for j in range(self.pad_size, width, self.seg_size):
                part = img_tensor[:, :,
                       (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
                       (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
                if padding > 0:
                    part = nn.ZeroPad2d(padding)(part)
                patches.append(part)
        return patches

    def merge_patches(self, patches):
        out = torch.zeros(1, 3, self.height * self.scale_factor, self.width * self.scale_factor,
                          device=patches[0].device)
        img_tensors = copy.copy(patches)
        rem = self.pad_size * 2

        pad_size = self.scale_factor * self.pad_size
        seg_size = self.scale_factor * self.seg_size
        height = self.scale_factor * self.height
        width = self.scale_factor * self.width
        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                part = img_tensors.pop(0)
                part = part[:, :, rem:-rem, rem:-rem]
                _, _, p_h, p_w = part.size()
                out[:, :, i:i + p_h, j:j + p_w] = part
        out = out[:, :, rem:-rem, rem:-rem]
        return out.squeeze(0)


def psnr(predict, target):
    with torch.no_grad():
        mse = F.mse_loss(predict, target)
        return -10 * torch.log10(mse)


@torch.no_grad()
def calc_baseline(loader, scale_factor=2, use_cuda=torch.cuda.is_available(),
                  method='bicubic', criterion=nn.L1Loss()):
    loss, psnr_, ssim = WeightedAveragedMetric(), WeightedAveragedMetric(), WeightedAveragedMetric()
    with tqdm(loader, desc='Baseline') as bar:
        for inputs, targets in bar:
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            resized = F.interpolate(inputs, scale_factor=scale_factor, mode=method)

            size = len(targets)
            loss.update(criterion(resized, targets).item(), size)
            psnr_.update(psnr(resized, targets).item(), size)
            ssim.update(pytorch_msssim.ms_ssim(resized, targets, data_range=1).item(), size)

            bar.set_postfix({
                'loss': f'{loss.compute():.4g}',
                'psnr': f'{psnr_.compute():.2f}',
                'ssim': f'{ssim.compute():.4f}'
            })
