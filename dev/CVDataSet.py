import math
import os
from pathlib import Path

import cv2
import numpy as np
from cachetools import FIFOCache, cached
from torch.utils.data import Dataset
from tqdm import tqdm


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


class ImgLoader:

    def __init__(self):
        self.cache_dir = None
        self.data_dir = None

    def load_from_path(self, file_path: str = None):
        return ImgLoader.load_from_path_static(self.cache_dir, self.data_dir, file_path)

    @staticmethod
    @cached(cache=FIFOCache(maxsize=10))
    def load_from_path_static(cache_dir: Path = None, data_dir: Path = None, file_path: str = None):
        path__npy_ = cache_dir / f"{file_path}.npy"

        if os.path.exists(path__npy_):
            img_l = np.load(str(path__npy_))
            return img_l

        if not os.path.exists(path__npy_.parent):
            os.makedirs(path__npy_.parent)

        if os.path.isfile(data_dir / file_path):
            img_l = cv2.imread(str(data_dir / file_path), 0)
            np.save(str(path__npy_), img_l)
            return img_l

        if os.path.isdir(data_dir / file_path):
            img_l = []
            files, _ = os.listdir(data_dir / file_path)
            for file in tqdm(files):
                img_l.append(ImgLoader.load_from_path_static(cache_dir, data_dir / f"{file_path}/{file}", data_dir))

            img_l = np.stack(img_l)
            np.save(str(path__npy_), img_l)
            return img_l
