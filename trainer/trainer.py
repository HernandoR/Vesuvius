import numpy as np
import torch
import wandb
from base import BaseTrainer
from torchvision.utils import make_grid

from utils import inf_loop, MetricTracker


# TODO wandb: WARNING Step cannot be set when using syncing with tensorboard. Please log your step values as a metric such as 'global_step'
class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # self.train_metrics = MetricTracker('train/loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('valid/loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        keys = ['loss', *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = MetricTracker(*['train/' + key for key in keys], writer=self.writer)
        self.valid_metrics = MetricTracker(*['valid/' + key for key in keys], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            Step = (epoch - 1) * self.len_epoch + batch_idx

            self.writer.set_step(Step)

            self.train_metrics.update('train/loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update('train/' + met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                log = self.train_metrics.result()
                self.wandb.log(log, step=Step)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'valid/' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                Step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                self.writer.set_step(Step, 'valid')
                self.valid_metrics.update('valid/loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update('valid/' + met.__name__, met(output, target))

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                log = self.valid_metrics.result()
                wandb.log(log, step=Step)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
            # wandb donot need this, wandb.watch() will do this
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
