# In[1]:
import gc
import math
import os
import shutil
import socket
import ssl
import time

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from cachetools import FIFOCache, cached
from torch import nn
from torch.optim import lr_scheduler as LRS, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from Loggers import *
from Utils import *

ssl._create_default_https_context = ssl._create_unverified_context


def decide_paths():
    HOST = socket.gethostname()

    if HOST.endswith("cloudlab.us"):
        # is_kaggle = False
        HOST = "cloudlab"
    kaggle_run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
    if kaggle_run_type is None:
        # is_kaggle = False
        pass
    else:
        # is_kaggle = True
        HOST = "kaggle"
        print("Kaggle run type: {}".format(kaggle_run_type))

    is_test = False
    is_train = True

    is_to_submit = kaggle_run_type == "Batch"

    if HOST == "cloudlab":
        ROOT_DIR = Path("/local/Codes/Vesuvius").absolute()
        DATA_DIR = ROOT_DIR / "data" / "raw"
        OUTPUT_DIR = ROOT_DIR / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    elif HOST == "kaggle":
        ROOT_DIR = Path("/kaggle")
        DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
        OUTPUT_DIR = ROOT_DIR / "working" / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "input"
    else:
        ROOT_DIR = Path("../../").absolute()
        DATA_DIR = ROOT_DIR / "data" / "raw"
        OUTPUT_DIR = ROOT_DIR / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    CP_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    CACHE_DIR = OUTPUT_DIR / "cache"
    print(f"ROOT_DIR: {ROOT_DIR}")
    assert os.listdir(DATA_DIR) != [], "Data directory is empty"

    for p in [ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR, LOG_DIR, CACHE_DIR]:
        if os.path.exists(p) is False:
            os.makedirs(p)

    return {
        "ROOT_DIR": ROOT_DIR,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "CP_DIR": CP_DIR,
        "LOG_DIR": LOG_DIR,
        "CACHE_DIR": CACHE_DIR,
        "EXTERNAL_MODELS_DIR": EXTERNAL_MODELS_DIR,
        "HOST": HOST,
    }


def get_saved_model_path(cfg, fold=1):
    return str(Path(cfg['PATHS']['CP_DIR']) / f'{cfg["model_name"]}_{cfg["in_channels"]}_best.pth')

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


def get_config():
    PATHS = decide_paths()
    _cfg = yaml.safe_load(open("../default_config.yaml", "r"))
    PATHS = {k: str(v) for k, v in PATHS.items()}
    _cfg["PATHS"] = PATHS
    with open("config.yaml", "w") as f:
        yaml.dump(_cfg, f)
    setup_logging(PATHS["LOG_DIR"], log_config="logger_config.json")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    _cfg["device"] = device

    import wandb
    Logger.info('wandb imported')
    if _cfg["resume"] and os.path.exists(get_saved_model_path(_cfg)):
        Logger.info('resuming if possible')
        wandb.init(project=_cfg['comp_name'], name=_cfg['run_id'],
                   config=_cfg, dir=_cfg['PATHS']['LOG_DIR'], resume=True, notes='resumed')
    else:
        Logger.info('New_start')
        wandb.init(project=_cfg['comp_name'], name=_cfg['run_id'],
                   config=_cfg, dir=_cfg['PATHS']['LOG_DIR'], notes='new_start')

    wandb.config['train_aug_list'] = albu.Compose(
        get_aug_list(_cfg['tile_size'], _cfg['in_channels'], type='train')).to_dict()
    wandb.config['valid_aug_list'] = albu.Compose(
        get_aug_list(_cfg['tile_size'], _cfg['in_channels'], type='valid')).to_dict()
    USE_WANDB = True

    _cfg = wandb.config

    return _cfg


# In[2]:


Logger = get_logger(__name__)
cfg = get_config()



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

    if cfg["HOST"] == "kaggle":
        weight = None
    _model = CustomModel(cfg, weight)
    return _model, None


def get_transforms(mode, cfg):
    return albu.Compose(get_aug_list(cfg['tile_size'], cfg['in_channels'], type=mode))



def make_dataset(img_set_ids, mode='train', dataset_mode='train'):
    imgs = []
    masks = []
    labels = []
    for set_id in img_set_ids:
        imgs.append(f"{mode}/{set_id}/surface_volume")
        masks.append(f"{mode}/{set_id}/mask.png")
        labels.append(f"{mode}/{set_id}/inklabels.png")

    dataset = VesuviusDataset(
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


def get_scheduler(cfg, optimizer):
    scheduler_cosine = LRS.CosineAnnealingLR(
        optimizer, cfg["epochs"], eta_min=1e-7)
    # scheduler = GradualWarmupSchedulerV2(
    #     optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler_cosine


def scheduler_step(scheduler, avg_val_loss, epoch):
    # scheduler.step(epoch)
    scheduler.step()


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-6):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    assert all(np.unique(targets) == [0, 1]), f'mask.unique():{np.unique(targets)} != [0, 1]'
    # assert all(np.unique(preds) == [0, 1]), f'mask_pred.unique():{np.unique(preds)} != [0, 1]'
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

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
    # dice = _fbeta_numpy(label_gt, (label_pred >= th).astype(int), beta=0.5)
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

    con_mx = pd.DataFrame([[ctp, cfp], [cfn, ctn]], columns=['P', 'N'], index=['P', 'N'])

    rates = pd.DataFrame([recall, precision, accuracy], index=['recall', 'precision', 'accuracy'], columns=["value"])
    Logger.info(f'Confusion matrix: \n'
                f'{con_mx} \n'
                f'Rates: \n'
                f'{rates}')

    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th


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



from torch.cuda.amp import GradScaler, autocast


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=cfg["use_amp"])
    losses = AverageMeter()

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for step, data in pbar:
            images, masks, labels, positions = data

            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            with autocast(cfg["use_amp"]):
                y_preds = model(images).squeeze()
                labels = labels.squeeze()

                loss = criterion(y_preds, labels)
                assert loss > 0, f'input should be 0-1, but got: labels: {labels.min()}-{labels.max()}, y_preds: {y_preds.min()}-{y_preds.max()}'

            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["max_grad_norm"])

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
            # start_idx = step*cfg["valid_batch_size"]             # end_idx = start_idx + batch_size
            # print(positions)
            for i, (x1, y1, x2, y2) in enumerate(zip(*positions)):
                label_pred[y1:y2, x1:x2] += y_preds[i]
                label_count[y1:y2, x1:x2] += np.ones((y2 - y1, x2 - x1))

    Logger.info(f'mask_count_min: {label_count.min(initial=0)}')
    label_pred /= label_count + 1e-8
    return losses.avg, label_pred


import itertools


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
    tiled_img = np.pad(test_img, [[0, 0], [0, pad0], [0, pad1]], constant_values=0)
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
            Logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
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
             }
            , save_path)


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
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    return valid_mask_gt


def main():
    #    fold_configs = create_foldset()
    Logger.info(cfg)

    model, best_loss = build_model(cfg)
    best_loss = best_loss if best_loss is not None else np.inf
    model.to(cfg["device"])

    early_stopping = EarlyStopping(
        patience=cfg.PATIENCE, verbose=True
    )
    fold_configs = create_foldset()
    for epoch in range(cfg["epochs"] // len(fold_configs)):
        optimizer = AdamW(model.parameters(), lr=cfg["lr"])
        scheduler = get_scheduler(cfg, optimizer)

        for fold_config in fold_configs:
            train_loader = make_dataset(fold_config["train"], dataset_mode='train')
            valid_loader = make_dataset(fold_config["valid"], dataset_mode='valid')

            valid_mask_gt = valid_loader.dataset.get_gt(0)

            valid_mask_gt = preprocess_valid_mask_gt(valid_mask_gt, cfg["tile_size"])

            start_time = time.time()

            # train
            avg_loss = train_fn(train_loader, model, criterion, optimizer, cfg["device"])

            # eval
            avg_val_loss, label_pred = valid_fn(
                valid_loader, model, criterion, cfg["device"], valid_mask_gt)

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

    wandb.save(get_saved_model_path(cfg))
    wandb.finish()

if __name__ == "__main__":
    main()