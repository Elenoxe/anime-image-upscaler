from datetime import datetime

import pytorch_msssim
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid

import utils
from data import ImageFolder
from metrics import WeightedAveragedMetric
from models import IMDN
from ranger import Ranger
from trainer import Trainer
from utils import RandomNoise

NOISE_RANGES = [
    (0, 0),
    (5, 25),
    (25, 50)
]

dataset = 'Danbooru'
checkpoint = None

patch_size = 128
scale_factor = 2

lr = 1e-3
batch_size = 12
initial_epoch = 1
epochs = 150
noise_level = 0

save_threshold_psnr = 35

use_cuda = torch.cuda.is_available()

# model = CARN(scale_factor=scale_factor)
model = IMDN(scale_factor=scale_factor)

model_name = type(model).__name__
if checkpoint is not None:
    model.load_state_dict(torch.load(checkpoint))

if use_cuda:
    model.cuda()

preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

transform = transforms.Compose([
    RandomNoise(NOISE_RANGES[noise_level]),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ImageFolder(f'data/{dataset}/train', patch_size, scale_factor, random_crop=True,
                            preprocess=preprocess, transform=transform, target_transform=target_transform)
val_dataset = ImageFolder(f'data/{dataset}/val', patch_size, scale_factor, random_crop=False,
                          transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

optimizer = Ranger(model.parameters(), lr=lr)
criterion = nn.L1Loss()
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=8, min_lr=1e-6)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = SummaryWriter(log_dir=f'runs/{model_name}-{dataset}-'
                                     f'{scale_factor}x-level{noise_level}-{timestamp}/train', flush_secs=10)
val_writer = SummaryWriter(log_dir=f'runs/{model_name}-{dataset}-'
                                   f'{scale_factor}x-level{noise_level}-{timestamp}/val', flush_secs=10)


class ScalerTrainer(Trainer):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.loss = WeightedAveragedMetric()
        self.val_loss = WeightedAveragedMetric()
        self.psnr = WeightedAveragedMetric()
        self.val_psnr = WeightedAveragedMetric()
        self.ssim = WeightedAveragedMetric()
        self.val_ssim = WeightedAveragedMetric()

        self.best_psnr = 0
        self.best_ssim = 0

    def train_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        loss = criterion(outputs, targets)

        with torch.no_grad():
            if batch_idx + 1 == len(train_loader):
                indices = torch.randperm(inputs.shape[0])[:2]
                baseline = F.interpolate(inputs[indices], scale_factor=scale_factor, mode='bicubic')
                example = targets[indices], baseline, outputs[indices]
                grid = make_grid(torch.cat(example, dim=-1)).clamp(0, 1)
                train_writer.add_image('Random Images From Last Batch', grid, self.current_epoch)
            ssim = pytorch_msssim.ms_ssim(outputs, targets, data_range=1)
            psnr = utils.psnr(outputs, targets)

            size = len(targets)
            self.loss.update(loss.item(), size)
            self.psnr.update(psnr.item(), size)
            self.ssim.update(ssim.item(), size)

        return loss, {
            'loss': f'{self.loss.compute():.4g}',
            'psnr': f'{self.psnr.compute():.2f}',
            'ssim': f'{self.ssim.compute():.4f}'
        }

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        loss = criterion(outputs, targets)

        if self.current_epoch is not None and batch_idx + 1 == len(val_loader):
            baseline = F.interpolate(inputs, scale_factor=scale_factor, mode='bicubic')
            grid = make_grid(torch.cat([targets, baseline, outputs], dim=-1), nrow=2).clamp(0, 1)
            val_writer.add_image('Images From Last Batch', grid, self.current_epoch)

        ssim = pytorch_msssim.ms_ssim(outputs, targets, data_range=1)
        psnr = utils.psnr(outputs, targets)

        size = len(targets)
        self.val_loss.update(loss.item(), size)
        self.val_psnr.update(psnr.item(), size)
        self.val_ssim.update(ssim.item(), size)

        return {
            'val_loss': f'{self.val_loss.compute():.4g}',
            'psnr': f'{self.val_psnr.compute():.2f}',
            'ssim': f'{self.val_ssim.compute():.4f}'
        }

    def on_train_end(self):
        epoch = self.current_epoch

        loss = self.loss.compute()
        psnr = self.psnr.compute()
        ssim = self.ssim.compute()

        lr = optimizer.param_groups[0]['lr']

        train_writer.add_scalar('metrics/loss', loss, epoch)
        train_writer.add_scalar('metrics/psnr', psnr, epoch)
        train_writer.add_scalar('metrics/ssim', ssim, epoch)
        train_writer.add_scalar('lr', lr, epoch)

        self.loss.reset()
        self.psnr.reset()
        self.ssim.reset()

    def on_validation_end(self):
        epoch = self.current_epoch
        if epoch is not None:
            val_loss = self.val_loss.compute()
            val_psnr = self.val_psnr.compute()
            val_ssim = self.val_ssim.compute()

            scheduler.step(val_loss)

            val_writer.add_scalar('metrics/loss', val_loss, epoch)
            val_writer.add_scalar('metrics/psnr', val_psnr, epoch)
            val_writer.add_scalar('metrics/ssim', val_ssim, epoch)

            save_path = f'checkpoints/{model_name}-{dataset}-{scale_factor}x-level{noise_level}-{timestamp}' \
                        f'/{epoch}-{val_psnr:.2f}({val_ssim:.4f}).pth'
            if epoch == initial_epoch + epochs - 1:
                self.save(save_path)
            elif val_psnr > save_threshold_psnr:
                better = False
                if val_psnr >= self.best_psnr:
                    better = True
                    self.best_psnr = val_psnr
                if val_ssim > self.best_ssim:
                    better = True
                    self.best_ssim = val_ssim
                if better:
                    self.save(save_path)

        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()


trainer = ScalerTrainer(model, optimizer)
# trainer.fit(train_loader, val_loader, initial_epoch=initial_epoch, epochs=epochs)
