import itertools
import math

import numpy as np
from torch.utils.data import Dataset

from data_loader.image_cache_loader import ImageCacheLoader
from logger import Loggers

Logger = Loggers.get_logger(__name__)


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

    # Try to save memory
    __slots__ = (
        "image_sets",
        "cfg",
        "masks",
        "labels",
        "transform",
        "imageLoader",
        "patch_pos",

    )

    def __init__(self, cfg, image_sets=None, masks=None, labels=None, transform=None):
        self.image_sets = image_sets
        self.cfg = cfg
        self.masks = masks
        self.labels = labels
        self.transform = transform
        # self.type = mode
        self.imgLoader = None
        self.patch_pos = []
        self.preprocess()

    def preprocess(self):
        if self.imgLoader is None:
            self.imgLoader = ImageCacheLoader(
                cache_dir=self.cfg['cache_dir'],
                data_dir=self.cfg['data_dir'])

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

    def index_patch_id(self, idx) -> (int, int):

        # indexing to the right patch
        img_id = 0
        patch_id = idx
        for i, posit_list in enumerate(self.patch_pos):
            if patch_id < len(posit_list):
                img_id = i
                break
            else:
                patch_id -= len(posit_list)
        return img_id, patch_id

    def __getitem__(self, idx):

        img_id, patch_id = self.index_patch_id(idx)

        x1, y1 = self.patch_pos[img_id][patch_id]
        x2 = x1 + self.cfg['tile_size']
        y2 = y1 + self.cfg['tile_size']

        # noted that the images and labels are by numpy till being transformed
        image = self.imgLoader.load_from_path(self.image_sets[img_id], channel=self.cfg['in_channels'])
        image = image[y1:y2, x1:x2]

        # if img.shape[0] != self.cfg['tile_size'] or img.shape[1] != self.cfg['tile_size']:
        #     pad_h = self.cfg['tile_size'] - img.shape[0]
        #     pad_w = self.cfg['tile_size'] - img.shape[1]
        #     img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

        label = self.imgLoader.load_from_path(self.labels[img_id])
        label = label[y1:y2, x1:x2]

        if label.shape[0] == label.shape[1] == self.cfg['tile_size']:
            pass
        else:
            # pad label
            pad_h = self.cfg['tile_size'] - label.shape[0]
            pad_w = self.cfg['tile_size'] - label.shape[1]
            label = np.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # data = self.transform(image=img, mask=label)
        # label = data["mask"]

        # it should be considered, if we should transform
        # into tensor or not

        assert image is not None
        assert label is not None

        # Notice the axis movement in train example
        # dig if needed.
        data = self.transform(image=image, mask=label)
        image = data["image"]
        label = data["mask"]
        #
        # label = label.to(torch.float32)
        if label.max() >= 2:
            # Logger.info(f' label max {label.max()} > 2')
            label = label / 255
        if image.max() >= 2:
            Logger.info(f'image max {image.max()} > 2')
            image = image / 255

        return image, label
