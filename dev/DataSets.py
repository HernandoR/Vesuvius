import itertools
import math
import os
from pathlib import Path

import cv2
import numpy as np
from cachetools import FIFOCache, cached, LRUCache
from torch.utils.data import Dataset
from tqdm.auto import tqdm


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

            x1_num = math.ceil((mask.shape[1] - self.cfg['tile_size']) / self.cfg['stride']) + 1
            y1_num = math.ceil((mask.shape[0] - self.cfg['tile_size']) / self.cfg['stride']) + 1
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

        return image, mask, label, (x1, y1, x2, y2)


class CVDataSet(Dataset):

    def __init__(self, imgs, transforms, labels=None, data_type=None, crop_size=256):
        self.crop_size = crop_size
        self.imgs = imgs
        self.transforms = transforms
        self.labels = labels
        self.data_type = data_type

        self.cell_counts = []
        for img in self.imgs:
            cell_count = math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                img.shape[2] / self.crop_size
            )
            self.cell_counts.append(cell_count)

    def __len__(self):
        data_count = 0
        if self.data_type == "train":

            self.cell_id_maps = {}

            counter = 0
            for img_num, img in enumerate(self.imgs):

                cell_count = math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                    img.shape[2] / self.crop_size
                )
                for cell_id in range(cell_count):
                    h_num = cell_id // math.ceil(
                        self.labels[img_num].shape[1] / self.crop_size
                    )
                    w_num = cell_id - (
                            h_num
                            * math.ceil(self.labels[img_num].shape[1] / self.crop_size)
                    )

                    cropped_img = self.labels[img_num][
                                  h_num * self.crop_size: h_num * self.crop_size
                                                          + self.crop_size,
                                  w_num * self.crop_size: w_num * self.crop_size
                                                          + self.crop_size,
                                  ]

                    if cropped_img.sum() == 0:
                        continue

                    data_count += 1

                    self.cell_id_maps[counter] = (img_num, cell_id)
                    counter += 1

        else:

            for img in self.imgs:
                data_count += math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                    img.shape[2] / self.crop_size
                )
        return data_count

    def calc_img_num(self, idx):
        cum_cell_count = 0
        for i, cell_count in enumerate(self.cell_counts):
            cum_cell_count += cell_count
            if idx + 1 <= cum_cell_count:
                return i, idx - (cum_cell_count - cell_count)

    def __getitem__(self, idx):

        if self.data_type == "train":
            img_num, cell_id = self.cell_id_maps[idx]
        else:
            img_num, cell_id = self.calc_img_num(idx)

        target_img = self.imgs[img_num]
        # if self.data_type != "test":
        if self.data_type in ["train", "valid"]:
            target_label = self.labels[img_num]

        # print(target_label.shape)
        target_img = np.moveaxis(target_img, 0, 2)
        # target_label = np.moveaxis(target_label, 0, 2)

        h_num = cell_id // math.ceil(target_img.shape[1] / self.crop_size)
        w_num = cell_id - (h_num * math.ceil(target_img.shape[1] / self.crop_size))

        cropped_img = target_img[
                      h_num * self.crop_size: h_num * self.crop_size + self.crop_size,
                      w_num * self.crop_size: w_num * self.crop_size + self.crop_size,
                      ]

        if self.data_type in ["train", "valid"]:
            cropped_label = target_label[
                            h_num * self.crop_size: h_num * self.crop_size + self.crop_size,
                            w_num * self.crop_size: w_num * self.crop_size + self.crop_size,
                            ]
            augmented = self.transforms(image=cropped_img, mask=cropped_label)
            img = augmented["image"]
            img = np.moveaxis(img, 2, 0)
            mask = augmented["mask"]
        else:
            augmented = self.transforms(image=cropped_img)
            img = augmented["image"]
            img = np.moveaxis(img, 2, 0)
            mask = -1

        return img, mask / 255



class ImageDataset(Dataset):
    def __init__(self, img_list, mask_list, transforms, label_list=None, data_type=None, img_loader=None,
                 crop_size=256):
        self.img_list = img_list
        self.transforms = transforms
        self.mask_list = mask_list
        self.label_list = label_list
        self.data_type = data_type
        self.crop_size = crop_size
        self.imgLoader = img_loader
        self.positions = []
        self.preprocess()

    def preprocess(self):
        # TODO:
        # This should pass the mask rather than the image

        for img_idx, mask_path in enumerate(self.mask_list):
            mask = self.imgLoader.load_from_path(mask_path)
            positions = self.find_non_masked_positions(mask, chop_size=self.crop_size)
            self.positions.append(positions)

    @staticmethod
    def find_non_masked_positions(mask, chop_size=256):
        # Pad the array with black edges to make its size divisible by 256
        height, width = mask.shape[:2]
        pad_height = (chop_size - height % chop_size) % chop_size
        pad_width = (chop_size - width % chop_size) % chop_size
        mask = np.pad(mask, ((0, pad_height), (0, pad_width)) + ((0, 0),) * (mask.ndim - 2), mode='constant')

        # Find non-zero positions in the array
        positions = []
        for n in range(0, mask.shape[0] - chop_size + 1, 128):
            for m in range(0, mask.shape[1] - chop_size + 1, 128):
                if np.sum(mask[n:n + chop_size, m:m + chop_size]) > 0:
                    positions.append((n, m))

        # # Adjust the positions to account for the black edges
        # positions = [(n - pad_height // 2, m - pad_width // 2) for n, m in positions]

        return positions

    def __len__(self):
        num = 0
        for pos in self.positions:
            num += len(pos)
        return num

    @cached(cache=LRUCache(maxsize=32))
    def __getitem__(self, index):
        # Determine the corresponding image and patch indices

        img_index = 0
        patch_index = index
        # TODO: consider a better way while loop
        for i in range(len(self.positions)):
            if patch_index >= len(self.positions[i]):
                patch_index -= len(self.positions[i])
                img_index += 1

        # Load image and mask
        img_path = self.img_list[img_index]
        mask_path = self.mask_list[img_index]

        pos_x, pos_y = self.positions[img_index][patch_index]

        img = self.imgLoader.load_from_path(img_path)
        mask = self.imgLoader.load_from_path(mask_path)

        # Get image dimensions
        img_width, img_height = img.shape[1], img.shape[2]

        # assert pos_x + 256 <= img_width, f"bad pos_x {pos_x} {img_width} on img {img_path},index {index}"
        # assert pos_y + 256 <= img_height, f"bad pos_y {pos_y} {img_height} on img {img_path},index {index}"

        img_patch = img[:, pos_x:pos_x + 256, pos_y:pos_y + 256]
        mask_patch = mask[pos_x:pos_x + 256, pos_y:pos_y + 256]

        # TODO: determine if we need to do this
        # img_patch = img_patch * mask_patch

        if self.data_type in ["train", "valid"]:
            assert self.label_list is not None, f'label list is None'

            label_path = self.label_list[img_index]
            label = self.imgLoader.load_from_path(label_path)
            label_patch = label[pos_x:pos_x + 256, pos_y:pos_y + 256]

            img_patch = np.moveaxis(img_patch, 0, 2)
            augmented = self.transforms(image=img_patch, mask=label_patch)
            img_patch = augmented["image"]
            img_patch = np.moveaxis(img_patch, 2, 0)
            label_patch = augmented["mask"]

        else:
            img_patch = np.moveaxis(img_patch, 0, 2)
            augmented = self.transforms(image=img_patch)
            img_patch = augmented["image"]
            img_patch = np.moveaxis(img_patch, 2, 0)
            label_patch = -1

        return img_patch, label_patch / 255


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
    def load_from_path_static(cache_dir: Path = None, data_dir: Path = None, file_path: str = None, channel=6):
        assert isinstance(file_path, (str, Path)), f"file path {file_path} is not a string or Path"
        if isinstance(file_path, Path):
            file_path = str(file_path)

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
