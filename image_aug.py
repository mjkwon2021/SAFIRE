import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import os
import cv2
from tqdm import tqdm
import random
import albumentations as A


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"],
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


def make_single_albu_post(key):
    if key == 'Resize':
        height_ratio = random.choice([np.random.uniform(0.5, 0.9), np.random.uniform(1.1, 1.5)])
        width_ratio = random.choice([np.random.uniform(0.5, 0.9), np.random.uniform(1.1, 1.5)])
        height = int(height_ratio * 2048)
        width = int(width_ratio * 2048)
        tf_str = f'A.{key}(height={str(height)},width={str(width)},p=1, interpolation=1)'
    elif 'Blur' in key:
        blur_limit = int(random.choice([3, 5, 7]))
        tf_str = f'A.{key}(blur_limit=({str(blur_limit)},{str(blur_limit)}),p=1)'
    elif key == 'Downscale':
        scale_min = random.uniform(0.7, 1)
        scale_max = scale_min
        tf_str = f'A.{key}(scale_min={str(scale_min)},scale_max={str(scale_max)},p=1, interpolation=1)'
    elif key == 'GaussNoise':
        var_limit = random.uniform(10, 40)
        tf_str = f'A.{key}(var_limit=({str(var_limit)},{str(var_limit)}),mean=0,p=1)'
    elif key == 'ISONoise':
        color_shift = random.uniform(0.01, 0.04)
        intensity = random.uniform(0.1, 0.5)
        tf_str = f'A.{key}(color_shift=({str(color_shift)},{str(color_shift)}),intensity=({str(intensity)},{str(intensity)}),p=1)'
    elif key == 'RandomBrightnessContrast':
        brightness_limit = random.uniform(-0.1, 0.1)
        contrast_limit = random.uniform(-0.1, 0.1)
        while abs(brightness_limit) < 0.05 and abs(contrast_limit) < 0.05:
            brightness_limit = random.uniform(-0.1, 0.1)
            contrast_limit = random.uniform(-0.1, 0.1)
        tf_str = f'A.{key}(brightness_limit=({str(brightness_limit)},{str(brightness_limit)}),contrast_limit=({str(contrast_limit)},{str(contrast_limit)}),p=1)'
    elif key == 'RandomGamma':
        gamma_limit = int(random.uniform(60, 150))
        while abs(gamma_limit - 100) < 10:
            gamma_limit = int(random.uniform(60, 150))
        tf_str = f'A.{key}(gamma_limit=({str(gamma_limit)},{str(gamma_limit)}),p=1)'
    elif key == 'CLAHE':
        clip_limit = random.uniform(1, 4)
        tile_grid_size = 2 ** random.choice(np.arange(1, 4))
        tf_str = f'A.{key}(clip_limit=({str(clip_limit)},{str(clip_limit)}),tile_grid_size=({str(tile_grid_size)},{str(tile_grid_size)}),p=1)'
    elif key == 'HueSaturationValue':
        hue_shift_limit = random.uniform(-10, 10)
        sat_shift_limit = random.uniform(-15, 15)
        val_shift_limit = random.uniform(-10, 10)
        while abs(hue_shift_limit) < 2.5 and abs(sat_shift_limit) < 2.5 and abs(val_shift_limit) < 2.5:
            hue_shift_limit = random.uniform(-10, 10)
            sat_shift_limit = random.uniform(-15, 15)
            val_shift_limit = random.uniform(-10, 10)
        tf_str = f'A.{key}(hue_shift_limit=({str(hue_shift_limit)},{str(hue_shift_limit)}),sat_shift_limit=({str(sat_shift_limit)},{str(sat_shift_limit)}),val_shift_limit=({str(val_shift_limit)},{str(val_shift_limit)}),p=1)'
    else:
        raise ValueError(f"Key is wrong. There's no key {key}.")

    # print(tf_str)
    return eval(tf_str)


key_list = [['Resize'], ['Blur', 'GaussianBlur', 'MedianBlur', 'MotionBlur'], ['Downscale'], ['GaussNoise', 'ISONoise'],
            ['RandomBrightnessContrast'], ['RandomGamma'], ['CLAHE'], ['HueSaturationValue']]
key_weight = [2, 1, 1, 1, 1, 1, 1, 1]


def weighted_sample_without_replacement(population, weights, k):
    chosen = []
    for _ in range(k):
        cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]
        total = cumulative_weights[-1]
        r = random.uniform(0, total)
        for i, cw in enumerate(cumulative_weights):
            if cw >= r:
                chosen.append(random.choice(population.pop(i)))
                weights.pop(i)
                break
    return chosen


def make_albu_post(num=1):
    keys = weighted_sample_without_replacement(key_list.copy(), key_weight.copy(), num)
    albu_list = []
    for key in keys:
        albu_list.append(make_single_albu_post(key))
    albu_list.append(A.Resize(height=1024, width=1024, p=1, interpolation=1))
    return A.Compose(albu_list)


def post_processing(fore_image, fore_mask):
    num_fore = random.choices([1, 2, 3, 4], weights=[4, 3, 2, 1], k=1)[0]
    albu_post_fore = make_albu_post(num=num_fore)
    albu_result_fore = albu_post_fore(image=fore_image, mask=fore_mask)
    albu_fore = albu_result_fore['image']
    albu_mask = albu_result_fore['mask']
    # albu_sum_fore = albu_result_summarize(albu_result_fore['replay'])
    return albu_fore, albu_mask
