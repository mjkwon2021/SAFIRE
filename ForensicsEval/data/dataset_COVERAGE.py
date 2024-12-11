"""
 Myung-Joon Kwon
 2023-07-26
"""

from ForensicsEval import project_config
from ForensicsEval.data.abstract import AbstractForgeryDataset

import os
from pathlib import Path
import numpy as np
from PIL import Image
import random
import cv2


class Dataset_COVERAGE(AbstractForgeryDataset):
    """
    directory structure
    COVERAGE (dataset_path["COVERAGE"] in project_config.py)
    ├── image
    │   ├── 1.tif
    │   └── 1t.tif ...
    └── mask
    """
    def __init__(self, im_list_file):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['COVERAGE']
        self.im_list_file = im_list_file
        with open(project_config.project_root / im_list_file, 'r') as f:
            self.im_list = [t.strip().split(',') for t in f.readlines()]

    def get_img_path(self, index):
        img_path = self.root_path / self.im_list[index][0]
        return img_path

    def get_mask_path(self, index):
        mask_path = self.root_path / self.im_list[index][1] if self.im_list[index][1] != "None" else None
        return mask_path

    def get_mask(self, index):
        mask_path = self.get_mask_path(index)
        if mask_path is None:
            return None  # mask = np.zeros((h, w))
        else:
            mask_img = Image.open(mask_path).convert('L')
            im_size = Image.open(self.get_img_path(index)).size
            mask_img = mask_img.resize(im_size, resample=Image.NEAREST)
            mask = np.array(mask_img)
            mask[mask <= 120] = 0
            mask[mask > 0] = 1
        return mask


