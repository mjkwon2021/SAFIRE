"""
 Myung-Joon Kwon
 2023-09-06
"""

from .. import project_config
from .abstract import AbstractForgeryDataset

import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


class Dataset_tampDPR(AbstractForgeryDataset):
    """
        directory structure:
        tampCOCO (dataset_path["tampDPR"] in project_config.py)
        ├── s0_m0_p0_c0_images
        │   ├── 000001.png
        │   └── ...
        ├── s0_m0_p0_c0_masks
        └ ...
        """
    def __init__(self, im_list_file, s=None, m=None, p=None, c=None):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['tampDPR']
        self.im_list_file = im_list_file
        with open(project_config.project_root / im_list_file, 'r') as f:
            self.im_list = [t.strip().split(',') for t in f.readlines()]
        if s is not None:
            self.im_list = [im for im in self.im_list if int(im[2])==s]
        if m is not None:
            self.im_list = [im for im in self.im_list if int(im[3])==m]
        if p is not None:
            self.im_list = [im for im in self.im_list if int(im[4])==p]
        if c is not None:
            self.im_list = [im for im in self.im_list if int(im[5])==c]
        self.smpc = f"{str(s) if s is not None else '-'}{str(m) if m is not None else '-'}{str(p) if p is not None else '-'}{str(c) if c is not None else '-'}"

    def __str__(self):
        return f"ForensicsEval.data.dataset_tampDPR.Dataset_tampDPR_{self.smpc}"
    def get_img_path(self, index):
        img_path = self.root_path / self.im_list[index][0]
        return img_path

    def get_mask_path(self, index):
        mask_path = self.root_path / self.im_list[index][1] if self.im_list[index][1] != "None" else None
        return mask_path

    def get_smpc_info(self, index):
        s, m, p, c = self.im_list[index][2:6]
        return int(s), int(m), int(p), int(c)


if __name__ == '__main__':
    random.seed(2024)
    dataset_name = 'tampDPR'

    # Creating dataset lists
    _root_path = project_config.dataset_paths[dataset_name]
    tamp_list = []
    tamp_train_list = []
    tamp_valid_list = []
    for type_dir in sorted(os.listdir(_root_path)):
        if os.path.isdir(_root_path / type_dir) and type_dir.endswith('images'):
            s, m, p, c, _ = type_dir.split("_")
            for i, file in enumerate(sorted(os.listdir(_root_path / type_dir))):
                mask_file = file
                line = f"{type_dir}/{file},{type_dir.replace('images', 'masks')}/{mask_file},{s[1]},{m[1]},{p[1]},{c[1]}"
                tamp_list.append(line)
                if i < 50:
                    tamp_valid_list.append(line)
                else:
                    tamp_train_list.append(line)


    with open(f"img_lists/{dataset_name}_tamp.txt", "w") as f:
        f.write('\n'.join(tamp_list) + '\n')

    with open(f"img_lists/{dataset_name}_tamp_train.txt", "w") as f:
        f.write('\n'.join(tamp_train_list) + '\n')
    with open(f"img_lists/{dataset_name}_tamp_valid.txt", "w") as f:
        f.write('\n'.join(tamp_valid_list) + '\n')

