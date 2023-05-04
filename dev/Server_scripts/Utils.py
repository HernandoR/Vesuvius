# In[1]:
import itertools
import math
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
from cachetools import FIFOCache, cached
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from Loggers import *


class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg
        self.th = None
        # if cfg["model_name"] == "Unet++":
        #     model_Conductor=getattr(smp, "UnetPlusPlus")
        # else:
        #     model_Conductor=getattr(smp, cfg["model_name"])
        # self.encoder = model_Conductor(
        self.encoder = smp.UnetPlusPlus(
            encoder_name=cfg['backbone'],
            encoder_weights=weight,
            in_channels=cfg['in_channels'],
            classes=cfg['target_size'],
            activation=None,

        )

    def forward(self, image):
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output


class VesuviusDataset(Dataset):
    """"
    Custom Dataset for loading images, masks, and labels
    params:
        image_sets: list of image paths
        masks: list of mask paths
        labels: list of label paths
        transform: albumentations transform
        mode: train, valid, or test
    """

    def __init__(self, image_sets, cfg, masks=None, labels=None, transform=None, mode="train"):
        self.image_sets = image_sets
        self.cfg = cfg
        self.masks = masks
        self.labels = labels
        self.transform = transform
        self.type = mode
        self.imgLoader = None
        self.patch_pos = []
        self.preprocess()

    def preprocess(self):
        if self.imgLoader is None:
            self.imgLoader = ImgLoader(
                cache_dir=self.cfg['PATHS']['CACHE_DIR'],
                data_dir=self.cfg['PATHS']['DATA_DIR'])

        for mask in self.masks:
            # mask may be path like or numpy array
            # if isinstance(mask, (str, Path)):
            mask = self.imgLoader.load_from_path(mask)

            x1_num = math.ceil((mask.shape[1] - self.cfg['tile_size']) / self.cfg['stride']) + 1
            y1_num = math.ceil((mask.shape[0] - self.cfg['tile_size']) / self.cfg['stride']) + 1
            posits = []
            for x, y in itertools.product(range(x1_num), range(y1_num)):
                x, y = x * self.cfg['stride'], y * self.cfg['stride']
                if mask[y:y + self.cfg['tile_size'], x:x + self.cfg['tile_size']].sum() > 0:
                    posits.append((x, y))
            self.patch_pos.append(posits)

        # self.patch_pos = np.stack(self.patch_pos)

    def get_gt(self, img_idx):
        return self.imgLoader.load_from_path(self.labels[img_idx])

    def get_mask(self, img_idx):
        return self.imgLoader.load_from_path(self.masks[img_idx])

    def __len__(self):
        return sum([len(posit) for posit in self.patch_pos])

    def __getitem__(self, idx):
        # x1, y1, x2, y2 = self.xyxys[idx]
        img_id = 0
        patch_id = idx
        for i, posit_list in enumerate(self.patch_pos):
            if patch_id < len(posit_list):
                img_id = i
                break
            else:
                patch_id -= len(posit_list)

        # x1_num, y1_num = self.patch_pos[img_id]
        # x1 = (patch_id % x1_num) * self.cfg['stride']
        # y1 = (patch_id // x1_num) * self.cfg['stride']
        x1, y1 = self.patch_pos[img_id][patch_id]
        x2 = x1 + self.cfg['tile_size']
        y2 = y1 + self.cfg['tile_size']

        img = self.imgLoader.load_from_path(self.image_sets[img_id], channel=self.cfg['in_channels'])
        mask = self.imgLoader.load_from_path(self.masks[img_id])

        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        if img.shape[0] != self.cfg['tile_size'] or img.shape[1] != self.cfg['tile_size']:
            pad_h = self.cfg['tile_size'] - img.shape[0]
            pad_w = self.cfg['tile_size'] - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        if self.type in ["train", "valid"]:
            label = self.imgLoader.load_from_path(self.labels[img_id])
            label = label[y1:y2, x1:x2]
            if label.shape[0] != self.cfg['tile_size'] or label.shape[1] != self.cfg['tile_size']:
                pad_h = self.cfg['tile_size'] - label.shape[0]
                pad_w = self.cfg['tile_size'] - label.shape[1]
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

            data = self.transform(image=img, mask=mask, label=label)
            label = data["label"].astype(np.float32)
        else:
            label = -1
            data = self.transform(image=img, mask=mask)

        image = data["image"]
        mask = data["mask"]

        return image, mask, label / 255, (x1, y1, x2, y2)


class ImgLoader:

    def __init__(self, cache_dir: Path = None, data_dir: Path = None):
        self.cache_dir = cache_dir
        self.data_dir = data_dir

    def load_from_path(self, file_path: str = None, channel=6, tile_size=224):
        ori_img = ImgLoader.load_from_path_static(self.cache_dir, self.data_dir, file_path, channel=channel)
        # pad_h = (tile_size - ori_img.shape[1] % tile_size) % tile_size
        # img = np.pad(ori_img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        # pad_w = (tile_size - ori_img.shape[2] % tile_size) % tile_size

        return ori_img

    @staticmethod
    @cached(cache=FIFOCache(maxsize=10))
    def load_from_path_static(cache_dir: str = None, data_dir: str = None, file_path: str = None, channel=6):
        assert isinstance(file_path, (str, Path)), f"file path {file_path} is not a string or Path"
        if isinstance(file_path, Path):
            file_path = str(file_path)
        cache_dir = Path(cache_dir)
        data_dir = Path(data_dir)
        assert not file_path.endswith('npy'), f"file path {file_path} is a npy file"
        assert os.path.exists(data_dir / file_path), f"file path {file_path} does not exist"

        path__npy_ = cache_dir / f"{file_path}.npy"

        if os.path.exists(path__npy_):
            img_l = np.load(str(path__npy_), allow_pickle=True)
            assert img_l is not None, f"Cached file {path__npy_} is None"
            return img_l

        if not os.path.exists(path__npy_.parent):
            os.makedirs(path__npy_.parent)

        if os.path.isfile(data_dir / file_path):
            img_l = cv2.imread(str(data_dir / file_path), 0)
            assert img_l is not None, f"Image file {data_dir / file_path} is None"
            np.save(str(path__npy_), img_l)
            return img_l

        if os.path.isdir(data_dir / file_path):
            path__npy_ = cache_dir / f"{file_path}_{channel}.npy"
            if os.path.exists(path__npy_):
                img_l = np.load(str(path__npy_), allow_pickle=True)
                assert img_l is not None, f"Cached file {path__npy_} is None"
                return img_l

            img_l = []
            files = os.listdir(data_dir / file_path)
            mid = len(files) // 2
            start = mid - channel // 2
            end = mid + channel // 2
            assert start >= 0, f"start {start} is less than 0"
            assert end <= len(files), f"end {end} is greater than {len(files)}"

            files = files[start:end]
            for file in tqdm(files):
                img_l.append(ImgLoader.load_from_path_static(cache_dir, data_dir, f"{file_path}/{file}"))

            img_l = np.stack(img_l, axis=2)
            np.save(str(path__npy_), img_l)
            return img_l


class AverageMeter(object):
    """Computes and stores the average and current value"""

    # if imported wandb, use wandb.log
    #

    def __init__(self, mode="train"):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.mode = mode

        self.reset()
        # self.use_wandb = (importlib.util.find_spec('wandb') is not None)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
