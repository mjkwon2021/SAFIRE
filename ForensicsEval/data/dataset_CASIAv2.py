"""
 Myung-Joon Kwon
 2023-07-25
"""

from .. import project_config
from .abstract import AbstractForgeryDataset

import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


class Dataset_CASIAv2(AbstractForgeryDataset):
    """
        directory structure:
        CASIA (dataset_path["CASIA"] in project_config.py)
        ├── CASIA 1.0 dataset (download: https://github.com/CauchyComplete/casia1groundtruth)
        │   ├── Au (un-zip it)
        │   └── Modified TP (un-zip it)
        ├── CASIA 1.0 groundtruth
        │   ├── CM
        │   └── Sp
        ├── CASIA 2.0 (download: https://github.com/CauchyComplete/casia2groundtruth)
        │   ├── Au
        │   └── Tp
        └── CASIA 2 Groundtruth  => Run renaming script in the excel file located in the above repo.
                                Plus, rename "Tp_D_NRD_S_N_cha10002_cha10001_20094_gt3.png" to "..._gt.png"
        """
    def __init__(self, im_list_file):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['CASIA']
        self.im_list_file = im_list_file
        with open(project_config.project_root / im_list_file, 'r') as f:
            self.im_list = [t.strip().split(',') for t in f.readlines()]

    def get_img_path(self, index):
        img_path = self.root_path / self.im_list[index][0]
        return img_path

    def get_mask_path(self, index):
        mask_path = self.root_path / self.im_list[index][1] if self.im_list[index][1] != "None" else None
        return mask_path
