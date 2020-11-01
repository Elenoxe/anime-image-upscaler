import abc
import os
from collections.abc import Iterable
from typing import Optional

import torch
from tqdm import tqdm


class Trainer(abc.ABC):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.current_epoch = None

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    @abc.abstractmethod
    def train_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def fit(self, train_loader, val_loader=None, epochs=1, initial_epoch=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'):
        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.current_epoch = epoch
            self.on_epoch_start(epoch)

            self.model.train()
            self.on_train_start()

            bar = tqdm(train_loader, desc=f'Epoch {epoch}/{initial_epoch + epochs - 1} Train')
            for batch_idx, (inputs, targets) in enumerate(bar):
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()

                train_step_rt = self.train_step((inputs, targets), batch_idx)
                train_metrics = None
                if isinstance(train_step_rt, Iterable):
                    loss, train_metrics = train_step_rt
                else:
                    loss = train_step_rt

                loss.backward()
                self.optimizer.step()

                if train_metrics is not None:
                    postfix = {}
                    for name, metric in train_metrics.items():
                        postfix[name] = metric
                    bar.set_postfix(postfix)
            bar.close()

            self.on_train_end()

            if val_loader is not None:
                self.eval(val_loader, f'Epoch {epoch}/{initial_epoch + epochs - 1} Eval')

            self.on_epoch_end(epoch)

        self.current_epoch = None

    @torch.no_grad()
    def eval(self, val_loader, desc='Eval', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model.eval()
        self.on_validation_start()
        with tqdm(val_loader, desc=desc) as bar:
            for batch_idx, (inputs, targets) in enumerate(bar):
                inputs, targets = inputs.to(device), targets.to(device)

                val_metrics: Optional[dict] = self.validation_step((inputs, targets), batch_idx)
                if val_metrics is not None:
                    postfix = {}
                    for name, metric in val_metrics.items():
                        postfix[name] = metric
                    bar.set_postfix(postfix)
        self.on_validation_end()

    def save(self, path, state_dict_only=True, extras=None):
        save_dir = os.path.dirname(path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        model = self.model.state_dict() if state_dict_only else self.model
        if extras is not None:
            checkpoint = {'model': model}
            checkpoint.update(extras)
        else:
            checkpoint = model
        torch.save(checkpoint, path)
