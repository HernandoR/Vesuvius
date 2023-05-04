#!/usr/bin/env python
# coding: utf-8

# ## summary

# In[1]:


from matplotlib import pyplot as plt
import importlib
import itertools
from torch.cuda.amp import GradScaler, autocast
import gc
import importlib.util
import math
import os
import shutil
import sys
import time
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler as LRS, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ## Configs
# ### Env detection

# In[2]:


import socket

HOST = socket.gethostname()
if HOST.endswith("cloudlab.us"):
    is_kaggle = False
    HOST = "cloudlab"

# noinspection PyUnresolvedReferences
# is_kaggle = _dh == ["/kaggle/working"]
is_kaggle = False

is_test = False

is_train = True

is_to_submit = False
Model_Proto = "Unet++"
exp_id = f"{Model_Proto}_2.5d_Unimodel"


# ### Auto settings

# In[3]:


date_time = time.strftime("%Y%m%d_%H%M%S")
run_id = exp_id + "_" + date_time
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

if is_test:
    is_train = False
    is_to_submit = False
    exp_id = exp_id + "_test"
    run_id = exp_id + "_" + date_time

USE_WANDB = False

exp_id, HOST, is_kaggle


# In[4]:

if HOST == "cloudlab":
    ROOT_DIR = Path("/local/Codes/Vesuvius").absolute()
    DATA_DIR = ROOT_DIR / "data" / "raw"
    OUTPUT_DIR = ROOT_DIR / "saved"
    CP_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = ROOT_DIR / "saved" / "logs"
    CACHE_DIR = ROOT_DIR / "saved" / "cache"
    EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

elif is_kaggle:
    ROOT_DIR = Path("/kaggle")
    DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
    OUTPUT_DIR = ROOT_DIR / "working"
    CP_DIR = OUTPUT_DIR / "ink-model"
    LOG_DIR = OUTPUT_DIR / "saved" / "logs"
    CACHE_DIR = OUTPUT_DIR / "saved" / "cache"
    EXTERNAL_MODELS_DIR = ROOT_DIR / "input"
else:
    ROOT_DIR = Path("../../").absolute()
    DATA_DIR = ROOT_DIR / "data" / "raw"
    OUTPUT_DIR = ROOT_DIR / "saved"
    CP_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = ROOT_DIR / "saved" / "logs"
    CACHE_DIR = ROOT_DIR / "saved" / "cache"
    EXTERNAL_MODELS_DIR = ROOT_DIR / "model"


print(f"ROOT_DIR: {ROOT_DIR}")

for p in [ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR, LOG_DIR, CACHE_DIR]:
    if os.path.exists(p) is False:
        os.makedirs(p)


# ### set Logger

# In[5]:


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))

    # remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# In[6]:

Logger = init_logger(log_file=str(LOG_DIR / run_id) + '.log')


# ### External models

# In[7]:


if not is_to_submit:
    # https://github.com/Cadene/pretrained-models.pytorch/issues/222
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
if is_kaggle:
    sys.path.append(str(EXTERNAL_MODELS_DIR / "segmentation-models-pytorch" /
                    "segmentation_models.pytorch-master"))
    sys.path.append(
        str(EXTERNAL_MODELS_DIR / "pretrainedmodels" / "pretrainedmodels-0.7.4"))
    sys.path.append(str(EXTERNAL_MODELS_DIR /
                    "efficientnet-pytorch" / "EfficientNet-PyTorch-master"))

    # noinspection PyUnresolvedReferences
    import segmentation_models_pytorch as smp

else:
    if importlib.util.find_spec("segmentation_models_pytorch") is None:
        get_ipython().system('conda install -y segmentation-models-pytorch')
    import segmentation_models_pytorch as smp
    # %%conda install -y -c conda-forge segmentation-models-pytorch


# ## configs

# In[8]:


class Config:
    comp_name = "vesuvius"
    # ============== path =============
    comp_dataset_path = DATA_DIR  # dataset path
    cache_dir = CACHE_DIR  # cache directory
    data_dir = DATA_DIR  # data directory
    LOG_DIR = LOG_DIR  # log directory
    Note_book = exp_id  # notebook name
    model_dir = CP_DIR  # model directory
    exp_name = exp_id  # experiment name
    run_id = run_id  # run id
    HOST = HOST  # host name

    # ============== pred target =============
    target_size = 1  # prediction target size

    # ============== model cfg =============
    model_name = f"{Model_Proto}"  # model name
    backbone = "se_resnext50_32x4d"  # model backbone
    in_channels = 16  # number of input channels
    resume = True  # resume training

    # ============== training cfg =============
    size = 224  # image size
    tile_size = 224  # tile size
    stride = 56  # tile stride

    batch_size = 16  # batch size
    use_amp = True  # use automatic mixed precision

    scheduler = "GradualWarmupSchedulerV2"  # learning rate scheduler
    epochs = 500  # number of training epochs
    PATIENCE = min(epochs // 2, 16)  # early stopping patience
    warmup_factor = 10  # warmup factor
    lr = 1e-5 / 10  # initial learning rate

    # ============== fold =============
    valid_id = 2  # validation fold id

    objective_cv = "binary"  # objective type
    metric_direction = "maximize"  # metric direction

    # ============== fixed =============
    pretrained = True  # use pre-trained weights
    inf_weight = "best"  # inference weight

    min_lr = 1e-6  # minimum learning rate
    weight_decay = 1e-6  # weight decay
    max_grad_norm = 1000  # maximum gradient norm

    print_freq = 50  # print frequency
    num_workers = 2  # number of workers
    seed = 3407  # random seed

    def __getitem__(self, item):
        return getattr(self, item)


_Config = Config()


# In[9]:


CONFIG = {}
for k in dir(_Config):
    if not k.startswith("__"):
        CONFIG[k] = getattr(_Config, k)

# check
for k in CONFIG.keys():
    if k not in CONFIG.keys():
        print(f"{k} is not in CONFIG")
    elif CONFIG[k] != CONFIG[k]:
        print(
            f"CONFIG {k} = {CONFIG[k]};is not equal to tmp {k} = {CONFIG[k]}")
    else:
        continue

for k in CONFIG.keys():
    if k not in CONFIG.keys():
        print(f"{k} is not in Config")
    else:
        continue

cfg = CONFIG


# In[10]:


# ============== augmentation =============
def get_aug_list(size, in_channels, type='train'):
    """
    type: train, valid
    return: list of albumentations

    in case of any further modification,
    one should use albu.Compose by themselves
    """
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        albu.Resize(size, size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=0.75),
        albu.ShiftScaleRotate(p=0.75),
        albu.OneOf([
            albu.GaussNoise(var_limit=(10.0, 50.0)),
            albu.GaussianBlur(),
            albu.MotionBlur(),
        ], p=0.4),
        albu.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        albu.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                           mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        albu.Normalize(
            mean=[0] * in_channels,
            std=[1] * in_channels
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        albu.Resize(size, size),
        albu.Normalize(
            mean=[0] * in_channels,
            std=[1] * in_channels
        ),
        ToTensorV2(transpose_mask=True),
    ]

    if type == 'train':
        return train_aug_list
    else:
        return valid_aug_list


# ## Helper functions

# In[11]:


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# def np_rle(img):
#     flat_img = np.where(img.flatten() , 1, 0).astype(np.uint8)
#     starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
#     ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
#     starts_ix = np.where(starts)[0] + 2
#     ends_ix = np.where(ends)[0] + 2
#     lengths = ends_ix - starts_ix
#     predicted_arr = np.stack([starts_ix, lengths]).T.flatten()
#     rle_str=np.array2string(predicted_arr.reshape(-1), separator=' ')
#     return rle_str[1:-1]


# ## Model

# In[12]:


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


def get_saved_model_path(cfg, fold=1):
    return str(cfg['model_dir'] / f'{cfg["model_name"]}_{cfg["in_channels"]}_best.pth')


def build_model(cfg, weight="imagenet"):
    Logger.info(f"model_name: {cfg['model_name']}")
    Logger.info(f"backbone: {cfg['backbone']}")
    if cfg['resume']:
        model_path = get_saved_model_path(cfg)
        if os.path.exists(model_path):
            Logger.info(f'load model from: {model_path}')
            _model = CustomModel(cfg, weight=None)
            loaded_model = torch.load(model_path)
            # print(loaded_model)
            _model.load_state_dict(loaded_model['model'])
            # best_loss = loaded_model['best_loss']
            # best_loss = None if loaded_model['best_loss'] is None else loaded_model['best_loss']
            best_loss = loaded_model['best_loss']
            th = loaded_model['th'] if 'th' in loaded_model else 0.5
            _model.th = th
            return _model, best_loss
        Logger.info(f'trained model not found')

    if is_kaggle:
        weight = None
    _model = CustomModel(cfg, weight)
    return _model, None


# ## DataSet

# In[13]:


def get_transforms(mode, cfg):
    return albu.Compose(get_aug_list(cfg['size'], cfg['in_channels'], type=mode))


if not is_kaggle:
    from DataSets import CustomDataset, ImgLoader
else:
    import itertools
    from cachetools import FIFOCache, cached

    class ImgLoader:

        def __init__(self, cache_dir: Path = None, data_dir: Path = None):
            self.cache_dir = cache_dir
            self.data_dir = data_dir

        def load_from_path(self, file_path: str = None, channel=6, tile_size=224):
            ori_img = ImgLoader.load_from_path_static(
                self.cache_dir, self.data_dir, file_path, channel=channel)
            # pad_h = (tile_size - ori_img.shape[1] % tile_size) % tile_size
            # img = np.pad(ori_img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            # pad_w = (tile_size - ori_img.shape[2] % tile_size) % tile_size

            return ori_img

        @staticmethod
        @cached(cache=FIFOCache(maxsize=10))
        def load_from_path_static(cache_dir: Path = None, data_dir: Path = None, file_path: str = None, channel=6):
            assert isinstance(file_path, (str, Path)
                              ), f"file path {file_path} is not a string or Path"
            if isinstance(file_path, Path):
                file_path = str(file_path)

            assert not file_path.endswith(
                'npy'), f"file path {file_path} is a npy file"
            assert os.path.exists(
                data_dir / file_path), f"file path {file_path} does not exist"

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
                assert end <= len(
                    files), f"end {end} is greater than {len(files)}"

                files = files[start:end]
                for file in files:
                    img_l.append(ImgLoader.load_from_path_static(
                        cache_dir, data_dir, f"{file_path}/{file}"))

                img_l = np.stack(img_l, axis=2)
                np.save(str(path__npy_), img_l)
                return img_l

    class CustomDataset(Dataset):
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
                    cache_dir=self.cfg['cache_dir'],
                    data_dir=self.cfg['data_dir'])

            for mask in self.masks:
                # mask may be path like or numpy array
                # if isinstance(mask, (str, Path)):
                mask = self.imgLoader.load_from_path(mask)

                x1_num = math.ceil(
                    (mask.shape[1] - self.cfg['tile_size']) / self.cfg['stride']) + 1
                y1_num = math.ceil(
                    (mask.shape[0] - self.cfg['tile_size']) / self.cfg['stride']) + 1
                posit = []
                for x, y in itertools.product(range(x1_num), range(y1_num)):
                    x, y = x * self.cfg['stride'], y * self.cfg['stride']
                    if mask[y:y + self.cfg['tile_size'], x:x + self.cfg['tile_size']].sum() > 0:
                        posit.append((x, y))
                self.patch_pos.append(posit)

            self.patch_pos = np.array(self.patch_pos)

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

            img = self.imgLoader.load_from_path(
                self.image_sets[img_id], channel=self.cfg['in_channels'])
            mask = self.imgLoader.load_from_path(self.masks[img_id])

            img = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]

            if img.shape[0] != self.cfg['tile_size'] or img.shape[1] != self.cfg['tile_size']:
                pad_h = self.cfg['tile_size'] - img.shape[0]
                pad_w = self.cfg['tile_size'] - img.shape[1]
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                             mode="constant", constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                              mode="constant", constant_values=0)

            if self.type in ["train", "valid"]:
                label = self.imgLoader.load_from_path(self.labels[img_id])
                label = label[y1:y2, x1:x2]
                if label.shape[0] != self.cfg['tile_size'] or label.shape[1] != self.cfg['tile_size']:
                    pad_h = self.cfg['tile_size'] - label.shape[0]
                    pad_w = self.cfg['tile_size'] - label.shape[1]
                    label = np.pad(label, ((0, pad_h), (0, pad_w)),
                                   mode="constant", constant_values=0)

                data = self.transform(image=img, mask=mask, label=label)
                label = data["label"].astype(np.float32)
            else:
                label = -1
                data = self.transform(image=img, mask=mask)

            image = data["image"]
            mask = data["mask"]

            return image, mask, label, (x1, y1, x2, y2)


# In[14]:


def make_dataset(img_set_ids, mode='train', dataset_mode='train'):
    imgs = []
    masks = []
    labels = []
    for set_id in img_set_ids:
        imgs.append(f"{mode}/{set_id}/surface_volume")
        masks.append(f"{mode}/{set_id}/mask.png")
        labels.append(f"{mode}/{set_id}/inklabels.png")

    dataset = CustomDataset(
        image_sets=imgs,
        cfg=cfg,
        masks=masks,
        labels=labels,
        mode=dataset_mode,
        transform=get_transforms(mode=dataset_mode, cfg=cfg))

    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=False)

    return loader


# ## Scheduler

# In[15]:


def get_scheduler(cfg, optimizer):
    scheduler_cosine = LRS.CosineAnnealingLR(
        optimizer, cfg["epochs"], eta_min=1e-7)
    # scheduler = GradualWarmupSchedulerV2(
    #     optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler_cosine


def scheduler_step(scheduler, avg_val_loss, epoch):
    # scheduler.step(epoch)
    scheduler.step()


# ## Assessments

# In[16]:


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-6):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    assert all(np.unique(targets) == [
               0, 1]), f'mask.unique():{np.unique(targets)} != [0, 1]'
    # assert all(np.unique(preds) == [0, 1]), f'mask_pred.unique():{np.unique(preds)} != [0, 1]'
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / \
        (beta_squared * c_precision + c_recall + smooth)

    return dice


def calc_fbeta(label_gt, label_pred):
    assert label_gt.shape == label_pred.shape, f'mask.shape:{label_gt.shape} != mask_pred.shape:{label_pred.shape}'
    label_gt = label_gt.astype(int).flatten()
    label_pred = label_pred.flatten()

    best_th = 0
    best_dice = 0
    # for th in np.array(range(10, 50 + 1, 5)) / 100:
    for th in np.linspace(10, 91, 9) / 100:
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        # print(label_pred.max())
        dice = fbeta_numpy(label_gt, (label_pred >= th).astype(int), beta=0.5)
        # Logger.info(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th

    # th=0.5
    # dice = fbeta_numpy(label_gt, (label_pred >= th).astype(int), beta=0.5)
    #
    # best_th = th
    # best_dice = dice

    # CTP,CFP, CTN, CFN
    ctp = label_pred[label_gt == 1].sum()
    cfp = label_pred[label_gt == 0].sum()
    ctn = (1 - label_pred)[label_gt == 0].sum()
    cfn = (1 - label_pred)[label_gt == 1].sum()
    # logger as matrix
    # Confused matrix
    recall = ctp / (ctp + cfn)
    precision = ctp / (ctp + cfp)
    accuracy = (ctp + ctn) / (ctp + ctn + cfp + cfn)

    con_mx = pd.DataFrame([[ctp, cfp], [cfn, ctn]], columns=[
                          'P', 'N'], index=['P', 'N'])

    rates = pd.DataFrame([recall, precision, accuracy], index=[
                         'recall', 'precision', 'accuracy'], columns=["value"])
    Logger.info(f'Confusion matrix: \n'
                f'{con_mx} \n'
                f'Rates: \n'
                f'{rates}')

    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th


# ## LOSS

# In[17]:


DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(
    mode='binary', log_loss=False, alpha=alpha, beta=beta)


def criterion(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f'y_pred.shape:{y_pred.shape} != y_true.shape:{y_true.shape}'
    # return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    return BCELoss(y_pred, y_true)
    # return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


# In[18]:


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


# ## Train and valid

# In[19]:


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CONFIG["use_amp"])
    losses = AverageMeter()

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for step, data in pbar:
            images, masks, labels, positions = data

            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            with autocast(CONFIG["use_amp"]):
                y_preds = model(images).squeeze()
                labels = labels.squeeze()

                loss = criterion(y_preds, labels)
                assert loss > 0, f'input should be 0-1, but got: labels: {labels.min()}-{labels.max()}, y_preds: {y_preds.min()}-{y_preds.max()}'

            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["max_grad_norm"])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, valid_mask_gt):
    label_pred = np.zeros(valid_mask_gt.shape)
    label_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter(mode='valid')

    with tqdm(enumerate(valid_loader), total=len(valid_loader)) as pbar:
        for step, (images, masks, labels, positions) in pbar:

            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            with torch.no_grad():
                y_preds = model(images).squeeze()
                labels = labels.squeeze()
                loss = criterion(y_preds, labels)
                assert loss > 0, f'input should be 0-1, but got: labels: {labels.min()}-{labels.max()}, y_preds: {y_preds.min()}-{y_preds.max()}'
            losses.update(loss.item(), batch_size)

            # make whole mask
            y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
            # start_idx = step*CONFIG["valid_batch_size"]             # end_idx = start_idx + batch_size
            # print(positions)
            for i, (x1, y1, x2, y2) in enumerate(zip(*positions)):
                label_pred[y1:y2, x1:x2] += y_preds[i]
                label_count[y1:y2, x1:x2] += np.ones((y2 - y1, x2 - x1))

    Logger.info(f'mask_count_min: {label_count.min()}')
    label_pred /= label_count + 1e-8
    return losses.avg, label_pred


# In[20]:


# TODO : modify test_fn
# seems it more like to use loader than single image
# consider tiled and stack?


def test_fn(test_img, model, device, mask_gt, tile_size=224):
    assert test_img.shape[1:3] == mask_gt.shape[0:2]
    model.eval()
    label_pred = np.zeros(mask_gt.shape)
    label_count = np.zeros(mask_gt.shape)

    ori_x, ori_y = mask_gt.shape
    if mask_gt.max() > 1:
        mask_gt = mask_gt / 255

    pad0 = (tile_size - ori_x % tile_size) % tile_size
    pad1 = (tile_size - ori_y % tile_size) % tile_size
    tiled_img = np.pad(test_img, [[0, 0], [0, pad0], [
                       0, pad1]], constant_values=0)
    tiled_mask = np.pad(mask_gt, [[0, pad0], [0, pad1]], constant_values=0)

    tiled_img = torch.from_numpy(tiled_img).float()
    tiled_mask = torch.from_numpy(tiled_mask).float()

    nx = (ori_x + pad0) // tile_size
    ny = (ori_y + pad1) // tile_size

    for x, y in itertools.product(range(nx), range(ny)):
        x1 = int(x * tile_size)
        x2 = int(x * tile_size + tile_size)
        y1 = int(y * tile_size)
        y2 = int(y * tile_size + tile_size)

        img = tiled_img[:, x1:x2, y1:y2]

        img = img.to(device)
        with torch.no_grad():
            y_preds = model(img).squeeze()

        label_pred[x1:x2, y1:y2] += y_preds > model.th
        label_count[x1:x2, y1:y2] += np.ones((tile_size, tile_size))

    label_pred /= label_count + 1e-8
    label_pred *= mask_gt

    return label_pred[:ori_x, :ori_y]


# ## EarlyStopping

# In[21]:


# TODO : modify EarlyStopping
# add detection of val_loss
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = None
        self.delta = delta

    def __call__(self, score, val_loss, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, val_loss, model)
            return

        if score > self.best_score + self.delta:
            self.save_checkpoint(score, val_loss, model)
            self.counter = 0
            return
        else:
            self.counter += 1
            Logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, val_loss, model):
        """Saves model when validation loss decrease."""

        if self.verbose:
            # Logger.info(
            #     f"Validation loss decreased ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ..."
            # )
            Logger.info(
                f"Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...")
        save_path = get_saved_model_path(cfg)
        if os.path.exists(save_path):
            shutil.move(save_path, save_path.replace('.pt', '-bkp.pt'))
        self.val_loss_min = val_loss
        self.best_score = score
        torch.save(
            {"model": model.state_dict(),
             "best_score": score,
             "best_loss": val_loss,
             "th": model.th
             }, save_path)


# ## Preparation
# ### prepare dataset

# In[22]:


def create_foldset():
    foldset = []
    foldset.append({
        'train': [1, 2],
        'valid': [3]
    })
    foldset.append({
        'train': [1, 3],
        'valid': [2]
    })
    foldset.append({
        'train': [2, 3],
        'valid': [1]
    })
    return foldset


def get_best_score(metric_direction):
    if metric_direction == 'minimize':
        return np.inf
    elif metric_direction == 'maximize':
        return -1
    else:
        return 0


def should_update_best_score(metric_direction, score, best_score):
    if metric_direction == 'minimize':
        return score < best_score
    elif metric_direction == 'maximize':
        return score > best_score


def preprocess_valid_mask_gt(valid_mask_gt, tile_size):
    if valid_mask_gt.max() > 1:
        Logger.info(f'valid_mask_gt.shape: {valid_mask_gt.shape} \n'
                    f'valid_mask_gt.max: {valid_mask_gt.max()} \n')
        valid_mask_gt = valid_mask_gt / 255

    pad0 = (tile_size - valid_mask_gt.shape[0] % tile_size)
    pad1 = (tile_size - valid_mask_gt.shape[1] % tile_size)
    valid_mask_gt = np.pad(
        valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    return valid_mask_gt


# ### wandb for trainning

# In[23]:


if is_train and not is_to_submit:
    if importlib.util.find_spec("wandb") is None:
        pass
    else:
        import wandb
        Logger.info('wandb imported')
        if cfg["resume"] and os.path.exists(get_saved_model_path(cfg)):
            Logger.info('resuming if possible')
            wandb.init(project=cfg['comp_name'], name=cfg['run_id'],
                       config=CONFIG, dir=LOG_DIR, resume=True, notes='resumed')
        else:
            Logger.info('New_start')
            wandb.init(project=cfg['comp_name'], name=cfg['run_id'],
                       config=CONFIG, dir=LOG_DIR)

        wandb.config['train_aug_list'] = albu.Compose(
            get_aug_list(cfg['size'], cfg['in_channels'], type='train')).to_dict()
        wandb.config['valid_aug_list'] = albu.Compose(
            get_aug_list(cfg['size'], cfg['in_channels'], type='valid')).to_dict()
        USE_WANDB = True


# ## Main

# In[24]:


#    fold_configs = create_foldset()
Logger.info(CONFIG)

model, best_loss = build_model(cfg)
best_loss = best_loss if best_loss is not None else np.inf
model.to(device)

if is_test:
    pass


if is_train:
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True
    )
    fold_configs = create_foldset()
    for epoch in range(CONFIG["epochs"]//len(fold_configs)):
        optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
        scheduler = get_scheduler(CONFIG, optimizer)

        for fold_config in fold_configs:
            train_loader = make_dataset(
                fold_config["train"], dataset_mode='train')
            valid_loader = make_dataset(
                fold_config["valid"], dataset_mode='valid')

            valid_mask_gt = valid_loader.dataset.get_gt(0)

            valid_mask_gt = preprocess_valid_mask_gt(
                valid_mask_gt, CONFIG["tile_size"])

            start_time = time.time()

            # train
            avg_loss = train_fn(train_loader, model,
                                criterion, optimizer, device)

            # eval
            avg_val_loss, label_pred = valid_fn(
                valid_loader, model, criterion, device, valid_mask_gt)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                Logger.info(f'best_loss: {best_loss:.4f}')
                torch.save(model.state_dict(), get_saved_model_path(cfg))

            scheduler_step(scheduler, avg_val_loss, epoch)

            best_dice, best_th = calc_cv(valid_mask_gt, label_pred)

            model.th = best_th

            # score = avg_val_loss
            score = best_dice

            elapsed = time.time() - start_time

            early_stopping(score, avg_val_loss, model)

            Logger.info(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            Logger.info(
                f'Epoch {epoch + 1} - avgScore: {score:.4f}')
            if USE_WANDB:
                wandb.log({
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "best_dice": best_dice,
                    "best_th": best_th,
                    "best_loss": best_loss,
                })
            if early_stopping.early_stop:
                Logger.info("Early stopping")
                break
        if early_stopping.early_stop:
            Logger.info("Early stopping")
            break


del model, optimizer, scheduler, early_stopping, train_loader, valid_loader, valid_mask_gt, label_pred
torch.cuda.empty_cache()
gc.collect()


# ## Prediction

# In[25]:


if USE_WANDB:
    wandb.finish()


# In[26]:


results = []
for id in ['a', 'b']:

    test_img = ImgLoader.load_from_path_static(
        cache_dir=cfg['cache_dir'],
        data_dir=cfg['data_dir'],
        file_path=f"test/{id}/surface_volume",
    )

    mask = ImgLoader.load_from_path_static(
        cache_dir=cfg['cache_dir'],
        data_dir=cfg['data_dir'],
        file_path=f"test/{id}/mask.png",
    )
    test_img = np.moveaxis(test_img, -1, 0)

    model, best_loss = build_model(cfg)
    model.to(device)
    label_pred = test_fn(test_img, model, device, mask, cfg['tile_size'])

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(test_img)
    ax[1].imshow(label_pred)
    ax[2].imshow(mask)
    ax[0].set_title('test_img')
    ax[1].set_title('label_pred')
    ax[2].set_title('mask')

    plt.show()
    results.append((id, label_pred))


# In[ ]:


sub = pd.DataFrame(results, columns=['Id', 'Predicted'])
sub.Id = sub.Id.asytpe(str)
sub


# In[ ]:


sample_sub = pd.read_csv(cfg['comp_dataset_path'] / 'sample_submission.csv')
sample_sub = pd.merge(sample_sub[['Id']], sub, on='Id', how='left')
sample_sub.to_csv('submission.csv', index=False)


# In[ ]:


# In[ ]:
