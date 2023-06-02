import os
from time import sleep
from pathlib import Path
from random import random

import cv2
import numpy as np
import psutil
from cachetools import FIFOCache, cached

from logger import Loggers

Logger = Loggers.get_logger(__name__)


class ImageCacheLoader:
    # get the available memory size
    _Memory_available = psutil.virtual_memory().available / 1024 / 1024 / 1024

    Logger.info(f"available memory size is {psutil.virtual_memory().available / 1024 / 1024 / 1024} GB")
    if _Memory_available < 20:
        cache_size = 16
    if _Memory_available > 100:
        Logger.info(f"seems no need for cache")
        cache_size = 128
    else:
        cache_size = 32

    def __init__(self, cache_dir: str = os.environ.get('TEMP', os.environ.get('TMP')), data_dir: str = None):
        assert data_dir is not None, f"data_dir is None"
        self.cache_dir = cache_dir
        self.data_dir = data_dir

    def load_from_path(self, file_path: str = None, channel=6, tile_size=224):
        ori_img = ImageCacheLoader.load_from_path_static(self.cache_dir, self.data_dir, file_path, channel=channel)
        # pad_h = (tile_size - ori_img.shape[1] % tile_size) % tile_size
        # img = np.pad(ori_img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        # pad_w = (tile_size - ori_img.shape[2] % tile_size) % tile_size

        return ori_img

    @staticmethod
    @cached(cache=FIFOCache(maxsize=cache_size))
    def load_from_path_static(cache_dir: str = None, data_dir: str = None, file_path: str = None, channel=6):
        assert isinstance(file_path, (str, Path)), f"file path {file_path} is not a string or Path"
        if isinstance(file_path, Path):
            file_path = str(file_path)

        cache_dir = Path(cache_dir)
        data_dir = Path(data_dir)

        assert not file_path.endswith('npy'), f"file path {file_path} is a npy file"
        assert os.path.exists(data_dir / file_path), f"file path {file_path} does not exist"

        path__npy_ = cache_dir / f"{file_path}_cache.npy"

        # load from cache
        if os.path.exists(path__npy_):
            img_l = np.load(str(path__npy_), allow_pickle=True)
            assert img_l is not None, f"Cached file {path__npy_} is None"
            return img_l

        while not os.path.exists(path__npy_.parent):
            sleep_time = random() * 0.1
            sleep(sleep_time)
            try:
                os.makedirs(path__npy_.parent)
            except FileExistsError:
                pass

        if os.path.isfile(data_dir / file_path):
            img_l = cv2.imread(str(data_dir / file_path), 0)
            assert img_l is not None, f"Image file {data_dir / file_path} is None"
            # img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            # img_l = img_l.astype(np.uint8)
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
            for file in files:
                img_l.append(ImageCacheLoader.load_from_path_static(cache_dir, data_dir, f"{file_path}/{file}"))

            img_l = np.stack(img_l, axis=2)
            np.save(str(path__npy_), img_l)
            return img_l
