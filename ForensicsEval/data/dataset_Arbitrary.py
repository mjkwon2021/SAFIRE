"""
 Myung-Joon Kwon
 2024-12-10
"""

from .. import project_config
from .abstract import AbstractForgeryDataset

import os


class Dataset_Arbitrary(AbstractForgeryDataset):
    def __init__(self):
        super().__init__()
        self.root_path = project_config.dataset_paths['Arbitrary']

        self.im_list = []
        for filenames in os.listdir(self.root_path):
            self.im_list.append([filenames, "None"])

    def get_img_path(self, index):
        img_path = self.root_path / self.im_list[index][0]
        return img_path

    def get_mask_path(self, index):
        mask_path = self.root_path / self.im_list[index][1] if self.im_list[index][1] != "None" else None
        return mask_path

    def get_filename(self, index):
        return self.im_list[index][0]
