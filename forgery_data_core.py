"""
Myung-Joon Kwon
2023-12-20
"""
import random

import albumentations as A
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import ForensicsEval
import cv2
from image_aug import post_processing
from robust_image_aug import robust_post_processing


def dilate_mask(mask, dilate_factor=3):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


class CoreDataset(Dataset):
    def __init__(self, datasets_list, mode='train', imsize=1024, augment_type=0, num_pairs=1, pp_type=None, pp_param=None, resize_mode=None, crop_prob=0):
        """
        :param datasets_list: list of datasets which inherit AbstractForgeryDataset
        :param mode: 'train', 'valid', 'test'
        :param imsize: 1024
        :param augment_type: 0
        :param num_pairs: the number of point pairs to sample. only used when mode='pair_random'
        """
        assert all(isinstance(dataset, ForensicsEval.data.abstract.AbstractForgeryDataset) for dataset in datasets_list)
        assert len(datasets_list) >= 1, "Must not be an empty list"
        self.datasets_list = datasets_list
        self.dataset_start_indices = [sum([len(ds) for ds in dss]) for dss in (datasets_list[:i] for i in range(len(datasets_list)))]
        self.mode = mode
        self.imsize = imsize
        self.augment_type = augment_type
        self.num_pairs = num_pairs
        self.pp_type = pp_type
        self.pp_param = pp_param
        assert resize_mode in [None, 'crop_prob', 'resize_when_large'], f'Invalid resize_mode: {resize_mode}'
        self.resize_mode = resize_mode
        self.crop_prob = crop_prob

    def __getitem__(self, idx):
        dataset_idx = max((i for i, x in enumerate(self.dataset_start_indices) if x <= idx), default=0)
        img_idx = idx - self.dataset_start_indices[dataset_idx]
        img, mask, mask_padinfo, img_path = self.load_item(dataset_idx, img_idx)
        # img: ndarray dtype=uint8, shape=(1024,1024,3)
        # mask: ndarray dtype=uint8, shape=(1024,1024)
        # mask_padinfo: ndarray dtype=uint8, shape=(1024,1024)

        # Safire test
        if self.mode in ["test_auto"]:
            mask_tensor = torch.tensor(mask[None, ...], dtype=torch.int64)
            img = img.astype('float32')

            # ignore padded area
            if mask_padinfo is not None:
                # repeat mask_padinfo along batch dimension to match with mask
                mask[mask_padinfo == 0] = -1
                img[mask_padinfo == 0] = 0

            return img, mask_tensor, str(img_path.resolve())

        # Safire pretrain : only tamp images
        if self.mode in ["im_mask"]:
            img = img.transpose((2, 0, 1))  # ndarray dtype=uint8, shape=(3,H,W)
            img = img.astype('float32')
            img_tensor = torch.tensor(img, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.int64)
            return img_tensor, mask_tensor, str(img_path.resolve())

        # Point prompt
        if self.mode in ["train", "valid", "one_random"]:
            target_value = np.random.choice(np.unique(mask))
            y_indices, x_indices = np.where((mask == target_value) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where(mask == target_value)
            index = np.random.randint(len(x_indices))
            point = np.array([x_indices[index], y_indices[index]])
            if target_value == 0:  # if auth is target, invert the mask
                mask = 1 - mask
            mask = mask[None, ...]
            point = point[None, ...]
        elif self.mode in ["mid", "one_mid"]:
            # use middle point
            if mask_padinfo is not None:
                y_indices, x_indices = np.where(mask_padinfo == 1)
                index = len(x_indices) // 2
                point = np.array([x_indices[index], y_indices[index]])
                mask = mask[None, ...]
            else:
                point = np.array([self.imsize//2, self.imsize//2])
                mask = mask[None, ...]
                point = point[None, ...]
        elif self.mode == "complete_random":
            y_indices, x_indices = np.where((mask == mask) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where(mask == mask)
            index = np.random.randint(len(x_indices))
            point = np.array([x_indices[index], y_indices[index]])
            mask = mask[None, ...]
            point = point[None, ...]
        elif self.mode in ["pair_random", "pair_random_cc"]:
            # use pair (one auth and one tamp)
            unique_list = np.unique(mask)
            points = []
            masks = []
            for point_pairs_i in range(self.num_pairs):
                if len(unique_list) == 2:
                    y_indices, x_indices = np.where((mask == 0) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where(mask == 0)
                    index = np.random.randint(len(x_indices))
                    point_0 = np.array([x_indices[index], y_indices[index]])
                    points.append(point_0)
                    masks.append(1 - mask)
                    y_indices, x_indices = np.where((mask == 1) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where(mask == 1)
                    index = np.random.randint(len(x_indices))
                    point_1 = np.array([x_indices[index], y_indices[index]])
                    points.append(point_1)
                    masks.append(mask)
                else:  # authentic image
                    y_indices, x_indices = np.where((mask == unique_list[0]) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where(mask == unique_list[0])
                    index = np.random.randint(len(x_indices))
                    point_0 = np.array([x_indices[index], y_indices[index]])
                    points.append(point_0)
                    masks.append(1 - mask if unique_list[0] == 0 else mask)
                    y_indices, x_indices = np.where((mask == unique_list[0]) & (mask_padinfo == 1)) if mask_padinfo is not None else np.where((mask == unique_list[0]))
                    index = np.random.randint(len(x_indices))
                    point_1 = np.array([x_indices[index], y_indices[index]])
                    points.append(point_1)
                    masks.append(1 - mask if unique_list[0] == 0 else mask)

            if self.mode == "pair_random_cc":  # connected component
                for i in range(len(masks)):
                    mask, point = masks[i], points[i]
                    cnt1, labels1 = cv2.connectedComponents(mask, connectivity=4)
                    cnt2, labels2 = cv2.connectedComponents(255 - mask, connectivity=4)
                    label3 = labels1 + labels2 * cnt1
                    point_value_mask = (label3 == label3[point[1], point[0]]).astype(np.uint8)
                    dilated_point_value_mask = dilate_mask(point_value_mask)
                    new_mask = np.array(mask, copy=True, dtype=int)  # I don't know why, but we need copy=True :P
                    new_mask[np.logical_not(np.isin(label3, label3[dilated_point_value_mask == 1]))] = -1
                    masks[i] = new_mask
            point = np.stack(points, axis=0)
            mask = np.stack(masks, axis=0)
        else:
            raise ValueError(f"Not supported value self.mode: {self.mode}.")

        # ignore padded area
        if mask_padinfo is not None:
            # repeat mask_padinfo along batch dimension to match with mask
            mask[np.repeat(mask_padinfo[None, ...], mask.shape[0], axis=0) == 0] = -1
            img[mask_padinfo == 0] = 0

        # ndarray to tensor
        img = img.transpose((2, 0, 1))  # ndarray dtype=uint8, shape=(3,H,W)
        img = img.astype('float')
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.int64)
        point_tensor = torch.tensor(point, dtype=torch.float32)

        return img_tensor, mask_tensor, point_tensor, str(img_path.resolve())

    def __len__(self):
        return self.dataset_start_indices[-1] + len(self.datasets_list[-1])

    def load_item(self, dataset_idx, img_idx):
        dataset = self.datasets_list[dataset_idx]
        img = dataset.get_img(img_idx)  # PIL
        W, H = img.size
        img = np.array(img)  # ndarray dtype=uint8, shape=(H,W,3)
        mask = dataset.get_mask(img_idx)

        if mask is None:
            mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        mask = mask.astype(np.uint8)

        # image augmentation
        if self.augment_type == 1 and np.random.rand() < 0.5:
            img, mask = post_processing(img, mask)
        elif self.augment_type == 2:
            img, mask = robust_post_processing(img, mask, self.pp_type, self.pp_param)

        # crop or resize
        mask_padinfo = None
        if self.resize_mode == 'crop_prob' and self.crop_prob > 0 and np.random.rand() < self.crop_prob:  # crop
            # if image size is smaller than self.imsize, we pad and crop.
            if H < self.imsize or W < self.imsize:
                pad_height = max(0, self.imsize - H)
                pad_width = max(0, self.imsize - W)
                img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='reflect')  # using reflect to avoid including new values

                # Crop. This is needed when only one dimension is smaller than self.imsize
                transform = A.Compose([
                    A.RandomCrop(width=self.imsize, height=self.imsize),
                ])
                transformed = transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']

                # Create new mask (mask_padinfo) to ignore padded area. 1: valid, 0: ignore
                mask_padinfo = np.ones_like(mask)
                mask_padinfo[H:, :] = 0
                mask_padinfo[:, W:] = 0
            # if image size is larger than self.imsize, we crop.
            else:
                transform = A.Compose([
                    A.RandomCrop(width=self.imsize, height=self.imsize),
                ])
                transformed = transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']

        elif self.resize_mode == 'resize_when_large' and max(H, W) < self.imsize:
            pad_height = max(0, self.imsize - H)
            pad_width = max(0, self.imsize - W)
            img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='reflect')

            # Create new mask (mask_padinfo) to ignore padded area. 1: valid, 0: ignore
            mask_padinfo = np.ones_like(mask)
            mask_padinfo[H:, :] = 0
            mask_padinfo[:, W:] = 0

        else:  # resize
            img = cv2.resize(img, (self.imsize, self.imsize))
            mask = cv2.resize(mask, (self.imsize, self.imsize), interpolation=cv2.INTER_NEAREST)

        return img, mask, mask_padinfo, dataset.get_img_path(img_idx)

    def shuffle_im_lists(self, random_seed=None):
        for i in range(len(self.datasets_list)):
            self.datasets_list[i].shuffle_im_list(random_seed)


if __name__ == '__main__':
    # from ForensicsEval.data import Dataset_COVERAGE, AbstractForgeryDataset
    # from ForensicsEval.data.dataset_COVERAGE import Dataset_COVERAGE
    # from ForensicsEval.data.abstract import AbstractForgeryDataset

    print(issubclass(ForensicsEval.data.Dataset_COVERAGE, ForensicsEval.data.AbstractForgeryDataset))
    a = ForensicsEval.data.Dataset_COVERAGE("data/img_lists/COVERAGE_tamp.txt")
    e = ForensicsEval.data.Dataset_tampCOCO("data/img_lists/bcm_COCO_tamp_train.txt"),
    b = ForensicsEval.data.Dataset_Carvalho("data/img_lists/Carvalho_auth.txt")
    c = ForensicsEval.data.Dataset_GRIP("data/img_lists/GRIP_auth.txt")
    D = CoreDataset([a,b,c,])
    print(D.dataset_start_indices)
    print(D[20])