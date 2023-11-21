from .transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from mmseg.datasets.pipelines.transforms import Resize, PhotoMetricDistortion


class SemiDataset(Dataset):
    def __init__(self, cfg, mode, id_path=None, nsample=None):
        self.name = cfg['dataset']
        self.root = os.path.expandvars(os.path.expanduser(cfg['data_root']))
        self.mode = mode
        self.size = cfg['crop_size']
        self.img_scale = cfg['img_scale']
        self.scale_ratio_range = cfg.get('scale_ratio_range', (0.5, 2.0))
        self.reduce_zero_label = cfg.get('reduce_zero_label', False)

        if isinstance(self.img_scale, list):
            self.img_scale = tuple(self.img_scale)
        self.labeled_photometric_distortion = cfg['labeled_photometric_distortion']

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            if id_path is None:
                id_path = 'splits/%s/val.txt' % self.name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        if self.reduce_zero_label:
            mask = np.array(mask)
            mask[mask == 0] = 255
            mask = mask - 1
            mask[mask == 254] = 255
            mask = Image.fromarray(mask)

        if self.mode == 'val':
            if self.img_scale is not None:
                res = Resize(img_scale=self.img_scale, min_size=512)(dict(
                    img=np.array(img),
                ))
                img = Image.fromarray(res['img'])
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.img_scale is not None:
            # print('Size before', img.size)
            res = Resize(img_scale=self.img_scale, ratio_range=self.scale_ratio_range)(dict(
                img=np.array(img),
                mask=np.array(mask),
                seg_fields=['mask']
            ))
            img = Image.fromarray(res['img'])
            mask = Image.fromarray(res['mask'])
            # print('Size after', mask.size)
        else:
            img, mask = resize(img, mask, self.scale_ratio_range)
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            if self.labeled_photometric_distortion:
                img = Image.fromarray(
                    PhotoMetricDistortion()({'img': np.array(img)[..., ::-1]})['img'][..., ::-1]
                )
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
