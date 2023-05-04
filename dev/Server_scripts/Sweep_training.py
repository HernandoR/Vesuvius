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




import socket

HOST = socket.gethostname()

if HOST.endswith('cloudlab.us'):
    HOST = 'cloudlab'
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


if not is_kaggle:
    ROOT_DIR = Path("../../").absolute()
    DATA_DIR = ROOT_DIR / "data" / "raw"
    OUTPUT_DIR = ROOT_DIR / "saved"
    CP_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = ROOT_DIR / "saved" / "logs"
    CACHE_DIR = ROOT_DIR / "saved" / "cache"
    EXTERNAL_MODELS_DIR = ROOT_DIR / "model"
else:
    ROOT_DIR = Path("/kaggle")
    DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
    OUTPUT_DIR = ROOT_DIR / "working"
    CP_DIR = OUTPUT_DIR / "ink-model"
    LOG_DIR = OUTPUT_DIR / "saved" / "logs"
    CACHE_DIR = OUTPUT_DIR / "saved" / "cache"
    EXTERNAL_MODELS_DIR = ROOT_DIR / "input"

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
    sys.path.append(str(EXTERNAL_MODELS_DIR / "segmentation-models-pytorch" / "segmentation_models.pytorch-master"))
    sys.path.append(str(EXTERNAL_MODELS_DIR / "pretrainedmodels" / "pretrainedmodels-0.7.4"))
    sys.path.append(str(EXTERNAL_MODELS_DIR / "efficientnet-pytorch" / "EfficientNet-PyTorch-master"))

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

## check
for k in CONFIG.keys():
    if k not in CONFIG.keys():
        print(f"{k} is not in CONFIG")
    elif CONFIG[k] != CONFIG[k]:
        print(f"CONFIG {k} = {CONFIG[k]};is not equal to tmp {k} = {CONFIG[k]}")
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