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


class Dataset_FantasticReality(AbstractForgeryDataset):
    def __init__(self, im_list_file):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['FantasticReality']
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
            return None
        mask = np.load(mask_path)['arr_0'].squeeze()
        mask[mask > 0] = 1
        return mask


if __name__ == '__main__':
    random.seed(2024)
    dataset_name = 'FantasticReality'
    tamp_valid_count = 100
    auth_valid_count = 100

    # Creating dataset lists
    _root_path = project_config.dataset_paths[dataset_name]

    # authentic
    auth_list = []
    prefix = "dataset/ColorRealImages"
    for file in os.listdir(_root_path / prefix):
        if file.endswith("jpg"):
            auth_list.append(f"{prefix}/{file},None")

    # tampered
    tamp_list = []
    img_prefix = "dataset/ColorFakeImages"
    mask_prefix = "dataset/SegmentationFake"
    for file in os.listdir(_root_path / img_prefix):
        if file.endswith("jpg"):
            tamp_list.append(f"{img_prefix}/{file},{mask_prefix}/{file.replace('.jpg', '.npz')}")

    with open(f"img_lists/{dataset_name}_auth.txt", "w") as f:
        f.write('\n'.join(auth_list)+'\n')
    with open(f"img_lists/{dataset_name}_tamp.txt", "w") as f:
        f.write('\n'.join(tamp_list)+'\n')
    print(len(auth_list), len(tamp_list))

    # valid
    random.shuffle(auth_list)
    random.shuffle(tamp_list)
    auth_valid_list = auth_list[:auth_valid_count]
    auth_train_list = auth_list[auth_valid_count:]
    tamp_valid_list = tamp_list[:tamp_valid_count]
    tamp_train_list = tamp_list[tamp_valid_count:]

    with open(f"img_lists/{dataset_name}_auth_valid.txt", "w") as f:
        f.write('\n'.join(auth_valid_list)+'\n')
    with open(f"img_lists/{dataset_name}_auth_train.txt", "w") as f:
        f.write('\n'.join(auth_train_list)+'\n')
    with open(f"img_lists/{dataset_name}_tamp_valid.txt", "w") as f:
        f.write('\n'.join(tamp_valid_list)+'\n')
    with open(f"img_lists/{dataset_name}_tamp_train.txt", "w") as f:
        f.write('\n'.join(tamp_train_list)+'\n')
    print(len(auth_valid_list), len(auth_train_list), len(tamp_valid_list), len(tamp_train_list), )

