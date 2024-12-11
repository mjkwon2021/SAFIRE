"""
 Myung-Joon Kwon
 2023-07-25
"""
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random


class AbstractForgeryDataset(ABC):
    def __init__(self):
        self.im_list = []
        self.max_num_img = None  # None or int

    def __len__(self):
        return len(self.im_list) if self.max_num_img is None else self.max_num_img

    @abstractmethod
    def get_img_path(self, index):
        pass

    def get_img(self, index):
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(self.get_img_path(index)).convert('RGB')
        return img

    @abstractmethod
    def get_mask_path(self, index):
        pass

    def get_mask(self, index):
        mask_path = self.get_mask_path(index)
        if mask_path is None:
            return None  # mask = np.zeros((h, w))
        else:
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img)
            mask[mask > 0] = 1
        return mask

    def repeat_im_list(self, times):
        new_im_list = self.im_list * times
        self.im_list = new_im_list

    def shuffle_im_list(self, random_seed=None):
        random.Random(random_seed).shuffle(self.im_list)

