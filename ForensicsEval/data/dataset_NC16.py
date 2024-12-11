"""
 Myung-Joon Kwon
 2023-07-26
"""

from .. import project_config
from .abstract import AbstractForgeryDataset

import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


class Dataset_NC16(AbstractForgeryDataset):
    def __init__(self, im_list_file):
        """
        :param im_list_file: path to txt file (from project_root). Format: img_file,mask_file
        """
        super().__init__()
        self.root_path = project_config.dataset_paths['NC16']
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
            # NC16 treats black as tampered
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img)
            mask[mask < 180] = 1
            mask[mask > 1] = 0
        return mask

if __name__ == '__main__':
    random.seed(2024)
    dataset_name = 'NC16'

    # Creating dataset lists
    _root_path = project_config.dataset_paths[dataset_name]

    with open(_root_path / "reference/manipulation/NC2016-manipulation-ref.csv") as f:
        lines = f.readlines()
    head = lines[0]
    column_names = head.split('|')
    index_of_IsTarget = column_names.index("IsTarget")
    index_of_ProbeFileName = column_names.index("ProbeFileName")
    index_of_ProbeMaskFileName = column_names.index("ProbeMaskFileName")
    index_of_BaseFileName = column_names.index("BaseFileName")
    index_of_IsRemoval = column_names.index("IsManipulationTypeRemoval")
    index_of_IsSplice = column_names.index("IsManipulationTypeSplice")
    index_of_IsCopyClone = column_names.index("IsManipulationTypeCopyClone")
    tamp_count = 0
    sp_count = 0
    cm_count = 0
    rm_count = 0
    auth_count = 0
    auth_list = []
    tamp_list = []
    for line in lines[1:]:
        items = line.split('|')
        if items[index_of_IsTarget] == 'Y':
            tamp_count += 1
            if items[index_of_IsRemoval] == 'Y':
                rm_count += 1
                manip_type_str = "RM"
            elif items[index_of_IsSplice] == 'Y':
                sp_count += 1
                manip_type_str = "SP"
            elif items[index_of_IsCopyClone] == 'Y':
                cm_count += 1
                manip_type_str = "CM"
            else:
                raise ValueError
            tamp_list.append(f"{items[index_of_ProbeFileName]},{items[index_of_ProbeMaskFileName]},{items[index_of_BaseFileName]},{manip_type_str}")
        else:
            auth_count += 1
            auth_list.append(f"{items[index_of_ProbeFileName]},None")
        # print(list(zip(column_names, items)))
    print(len(lines))
    print(auth_count, tamp_count)
    print(sp_count, cm_count, rm_count)


    # mask validation
    new_tamp_list = []
    from tqdm import tqdm
    for s in tqdm(tamp_list):
        im, mask = s.split(',')[:2]
        if not os.path.isfile(_root_path / mask):
            print("Skip:", im, mask)
            continue
        im_im = np.array(Image.open(_root_path / im))
        mask_im = np.array(Image.open(_root_path / mask))
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        new_tamp_list.append(s)
    tamp_list = new_tamp_list

    with open(f"img_lists/{dataset_name}_auth.txt", "w") as f:
        f.write('\n'.join(auth_list)+'\n')
    with open(f"img_lists/{dataset_name}_tamp.txt", "w") as f:
        f.write('\n'.join(tamp_list)+'\n')
    print(len(auth_list), len(tamp_list))
