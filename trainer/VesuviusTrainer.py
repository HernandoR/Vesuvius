import numpy as np
import torch
import wandb
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from base import BaseTrainer
from logger import Loggers
from utils import inf_loop, MetricTracker

Logger = Loggers.get_logger(__name__)


# TODO wandb: WARNING Step cannot be set when using syncing with tensorboard.
#  Please log your step values as a metric such as 'global_step'
class VesuviusTrainer(BaseTrainer):
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
        self.train_metrics = MetricTracker(*['train/' + key for key in keys], average_window=3)
        self.valid_metrics = MetricTracker(*['valid/' + key for key in keys], average_window=3)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        with tqdm(enumerate(self.data_loader,start=1), total=self.len_epoch) as pbar:
            for batch_idx, (data, target) in pbar:

                # data, target = data.astype(np.uint8), target.astype(np.uint8)
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data).squeeze()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                Step = (epoch - 1) * self.len_epoch + batch_idx

                self.train_metrics.update('train/loss', loss.item())

                for met in self.metric_ftns:
                    self.train_metrics.update('train/' + met.__name__, met(output, target))
                    # self.train_metrics.update('train/' + met.__name__, met(output_cpu, target_cpu))
                # del output_cpu, target_cpu

                if batch_idx % self.log_step == 0:
                    Logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    #
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    log = self.train_metrics.result()

                    B = data.shape[0]

                    input_tensor, output_tensor, target_tensor = data, output, target

                    output_tensor = (output_tensor.unsqueeze(1) > 0.5)
                    target_tensor = target_tensor.unsqueeze(1)
                    input_grid = make_grid(input_tensor, nrow=B)[0].cpu().numpy()
                    output_grid = make_grid(output_tensor, nrow=B)[0].cpu().numpy()
                    target_grid = make_grid(target_tensor, nrow=B)[0].cpu().numpy()
                    class_labels = {
                        0: "background",
                        1: "ink"
                    }
                    log.update({
                        'result': wandb.Image(
                            input_grid, masks={
                                "predictions": {"mask_data": output_grid.astype(np.uint8),
                                                "class_labels": class_labels},
                                "ground_truth": {"mask_data": target_grid.astype(np.uint8),
                                                 "class_labels": class_labels},
                            },
                        )
                    })
                    Logger.debug(f'Train Epoch: {epoch} batch_idx: {batch_idx},step: {Step}')

                    wandb.log(log, step=Step)
                    # del input_tensor, output_tensor, target_tensor, input_grid, output_grid, target_grid
                if batch_idx == self.len_epoch:
                    break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})

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
            with tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)) as pbar:
                for batch_idx, (data, target) in pbar:
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data).squeeze()
                    loss = self.criterion(output, target)

                    # Step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                    # self.writer.set_step(Step, 'valid')
                    self.valid_metrics.update('valid/loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update('valid/' + met.__name__, met(output, target))

                log = self.valid_metrics.result()

                B = data.shape[0]

                input_tensor, output_tensor, target_tensor = data, output, target

                output_tensor = (output_tensor.unsqueeze(1) > 0.5)
                target_tensor = target_tensor.unsqueeze(1)
                input_grid = make_grid(input_tensor, nrow=B)[0].cpu().numpy()
                output_grid = make_grid(output_tensor, nrow=B)[0].cpu().numpy()
                target_grid = make_grid(target_tensor, nrow=B)[0].cpu().numpy()
                class_labels = {
                    0: "background",
                    1: "ink"
                }
                log.update({
                    'result': wandb.Image(
                        input_grid, masks={
                            "predictions": {"mask_data": output_grid.astype(np.uint8),
                                            "class_labels": class_labels},
                            "ground_truth": {"mask_data": target_grid.astype(np.uint8),
                                             "class_labels": class_labels},
                        },
                    )
                })

        # make model parameters grid and add it to the wandb
        Step = epoch * self.len_epoch
        # # add histogram of model parameters to the wandb
        for name, p in self.model.named_parameters():
            # self.writer.add_histogram(name, p, bins='auto')
            log.update({f'parameters/{name}': wandb.Histogram(p.detach().cpu().numpy())})

        Logger.debug(f'valid Epoch: {epoch} batch_idx: {batch_idx},step: {Step}')
        # log.update({'input': wandb.Image(make_grid(data.cpu(), nrow=8, normalize=True))})
        wandb.log(log, step=Step, commit=True)

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
