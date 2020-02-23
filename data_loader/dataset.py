# coding : utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pathlib
import torch
import torch.utils.data as data
from PIL import Image
from .transforms import create_transform


class Dataset(data.Dataset):
    def __init__(self, dataset_file, transform=None, target_transform=None):
        self.dataset_file = dataset_file
        self.imgs = self.load_data()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except Exception as e:
            print('{}: {}'.format(img_path, e))
            return self[index+1]
        return img, target

    def __len__(self):
        return len(self.imgs)

    def load_data(self):
        file_name = self.dataset_file
        if not os.path.isfile(file_name):
            raise ValueError('invalid data list file')
        with open(file_name, 'r', encoding='utf-8') as file_reader:
            lines = file_reader.readlines()
        data_list = []
        for line in lines:
            data = line.strip().strip('\r\n').strip('\xef\xbb\xbf').split('\t')
            assert len(data) == 2, 'invalid annotation!'
            img_path = pathlib.Path(data[0])
            target = int(data[1])
            if img_path.exists() and img_path.stat().st_size > 0:
                data_list.append((str(img_path), target))
        return data_list


def create_loader(dataset, cfg, is_training=True):
    dataset.transform = create_transform(
        img_size=cfg.INPUT.IMG_SIZE,
        scale=cfg.INPUT.SCALE_TRAIN,
        is_training=is_training,
        color_jitter=(cfg.INPUT.BRIGHTNESS, cfg.INPUT.CONTRAST, cfg.INPUT.SATURATION, cfg.INPUT.HUE),
        auto_augment=cfg.INPUT.AUTO_AUGMENT,
        random_erasing=cfg.INPUT.RANDOM_ERASE_PROB,
        random_erasing_mode=cfg.INPUT.RANDOM_ERASE_MODE,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        crop_pct=cfg.INPUT.CROP_PCT_TEST)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=is_training,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_training,
    )
    return loader


