import json
import os
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from random import random
import random
import numpy as np
import pandas as pd
import torch
import yaml


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.safe_load(handle)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    # device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_saved_model_path(cfg, fold=1):
    return str(cfg['model_dir'] / f'{cfg["model_name"]}_{cfg["in_channels"]}_best.pth')


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class MetricTracker:
    def __init__(self, *keys, average_window=5):
        self.average_window = average_window
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._history = {}
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        self._history = {}

    def update(self, key, value, n=1):
        # if key not in self._history:
        #     self._history[key] = []
        # self._history[key].append((value, n))
        # if len(self._history[key]) > self.average_window:
        #     removed_value, N = self._history[key].pop(0)
        #     self._data.total[key] -= removed_value * N
        #     self._data.counts[key] -= N
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
