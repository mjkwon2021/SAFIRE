"""
 Myung-Joon Kwon
 2023-08-28
"""

from .. import project_config
from .abstract import AbstractForgeryDataset

import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


class Dataset_tampCOCO(AbstractForgeryDataset):
    """
        directory structure:
        tampCOCO (dataset_path["tampCOCO"] in project_config.py)
        ├── bcmc_images
        │   ├── 82_000000190150_aligned_Q100.jpg_000000190150_aligned_Q100.jpg_RT13.8_aligned_Q100.jpg
        │   └── ...
        ├── cm_images
        └ ...
        """
    def __init__(self, im_list_file):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['tampCOCO']
        self.im_list_file = im_list_file
        with open(project_config.project_root / im_list_file, 'r') as f:
            self.im_list = [t.strip().split(',') for t in f.readlines()]

    def get_img_path(self, index):
        img_path = self.root_path / self.im_list[index][0]
        return img_path

    def get_mask_path(self, index):
        mask_path = self.root_path / self.im_list[index][1] if self.im_list[index][1] != "None" else None
        return mask_path