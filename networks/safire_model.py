"""
Myung-Joon Kwon
2024-02-01
Use input normalization
"""
import os
import glob
import random
import monai
from os import makedirs
from os.path import join
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
import cv2
from matplotlib import pyplot as plt
import argparse
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union


class AdaptorSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder,
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # freeze image encoder (except adaptor)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            if name.startswith("adaptor") or name.startswith("hfp_") or name.startswith("feature_extractor"):
                param.requires_grad = True

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x.float()

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.
        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks[..., : input_size[0], : input_size[1]]
        # masks = F.interpolate(
        #     masks, original_size, mode="bilinear", align_corners=False
        # )   # WHY?

        return masks

    def forward(self, image, point_prompt, forward_type=0, upscale_output=True):
        image = self.preprocess(image)

        if forward_type==0:
            # do not compute gradients for pretrained img encoder if specified
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

            # do not compute gradients for pretrained prompt encoder
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=point_prompt,
                    boxes=None,
                    masks=None,
                )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )  # (B, 1, 256, 256)

            if upscale_output:
                ori_res_masks = F.interpolate(
                    low_res_masks,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                ori_res_masks = low_res_masks

            return ori_res_masks, iou_predictions  # was: low_res_masks

        elif forward_type==1:
            # do not compute gradients for pretrained img encoder if specified
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

            points, labels_torch = point_prompt  # (B, 2[auth,tamp], 2[x,y]), (B, 2[auth,tamp])

            group_of_ori_res_masks = []
            group_of_iou_predictions = []
            for i in range(points.shape[1]):
                # do not compute gradients for pretrained prompt encoder
                point_prompt = (points[:,i:i+1,:], labels_torch[:,i:i+1])
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=point_prompt,
                        boxes=None,
                        masks=None,
                    )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embedding,  # (B, 256, 64, 64)
                    image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )  # (B, 1, 256, 256)

                if upscale_output:
                    ori_res_masks = F.interpolate(
                        low_res_masks,
                        size=(image.shape[2], image.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    ori_res_masks = low_res_masks
                group_of_ori_res_masks.append(ori_res_masks)
                group_of_iou_predictions.append(iou_predictions)

            return group_of_ori_res_masks, group_of_iou_predictions
        else:
            raise NotImplementedError


