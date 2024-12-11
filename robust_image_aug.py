import random
import albumentations as A
import os
from PIL import Image
import sys
import numpy as np
import cv2


# post processing
def Gaussian_Blur(img_RGB, mask, kernel_size):
    # kernel_size = 4*random.randint(0,5)+3
    post_process = A.Compose([
        A.augmentations.GaussianBlur(blur_limit=(kernel_size, kernel_size), always_apply=True),
    ])
    aug = post_process(image=img_RGB, mask=mask)
    img_RGB = aug['image']
    mask = aug['mask']

    return img_RGB, mask


def Gaussian_Noise(img_RGB, stddev):
    # stddev = 4*random.randint(0,5)+3
    mean = 0
    # img_shape = img_RGB.shape[:2]
    noise = np.random.normal(mean, stddev, img_RGB.shape)
    img_RGB = np.clip(img_RGB.astype(np.float32) + noise.astype(np.float32), 0, 255)
    img_RGB = img_RGB.astype(np.uint8)
    return img_RGB


comp_dict = {q: A.ImageCompression(quality_lower=q-1, quality_upper=q, p=1) for q in range(50, 101)}


def JPEG_Compress(img_RGB, QF):
    comp_fn = comp_dict[QF]
    img_RGB = comp_fn(image=img_RGB)['image']

    return img_RGB


def Gamma_Correction(img_RGB, gamma):
    # gamma = 0.7+0.15*random.randint(0,4)
    img_RGB = (img_RGB / 255.0) ** (1.0 / gamma) * 255.0

    return img_RGB


def robust_post_processing(img, mask, pp_type, pp_param):
    assert isinstance(pp_type, str)
    if pp_type is None:
        ret_img = img
        ret_mask = mask
    elif pp_type == 'Gaussian_Blur':
        assert pp_param in range(3, 20, 4)
        ret_img, ret_mask = Gaussian_Blur(img, mask, kernel_size=pp_param)
    elif pp_type == 'Gaussian_Noise':
        assert pp_param in range(3, 24, 4)
        ret_img = Gaussian_Noise(img, stddev=pp_param)
        ret_mask = mask
    elif pp_type == 'JPEG_Compress':
        assert pp_param in range(50, 101, 10)
        ret_img = JPEG_Compress(img, QF=pp_param)
        ret_mask = mask
    elif pp_type == 'Gamma_Correction':
        assert pp_param in [0.7 + 0.15 * t for t in range(0, 5)]
        ret_img = Gamma_Correction(img, gamma=pp_param)
        ret_mask = mask
    else:
        raise ValueError(f"pp_type is wrong. There's no pp_type {pp_type}.")

    return ret_img, ret_mask