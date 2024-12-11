import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
import cv2
import torch.nn.functional as F

from safire_kmeans import kmeans
import random

from segment_anything.modeling import Sam
from segment_anything.predictor_safire import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

class SafirePredictor:
    def __init__(
        self,
        model,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
    ) -> None:
        assert (points_per_side is None) != (
                point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")
        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )

        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by confidence
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"],
            self.predictor.model.mask_threshold,
            self.stability_score_offset,
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        temp = data["masks"]
        data["masks"] = temp > self.predictor.model.mask_threshold
        data["pred_masks"] = F.sigmoid(temp).cpu().numpy()
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["pred_masks"] = uncrop_masks(data["pred_masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @torch.no_grad()
    def generate(self, image: np.ndarray, keep_features=False) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        self.predictor.set_image(image)
        points_scale = np.array((1024, 1024))[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, (1024, 1024), [0, 0, 1024, 1024], (1024, 1024)
            )
            data.cat(batch_data)
            del batch_data
        if not keep_features:
            self.predictor.reset_image()
        data.to_numpy()

        mask_data = data

        # Encode masks
        mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "pred_mask": mask_data["pred_masks"][idx],
            }
            curr_anns.append(ann)

        return curr_anns

    @torch.no_grad()
    def safire_predict(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], Any, List]:
        """
        Get the combined prediction for the given image.

        Arguments:
          image (np.ndarray): The image to predict, in HWC uint8 format.

        Returns:
           pred
        """
        curr_anns = self.generate(image, keep_features=True)

        # get avg features for each mask
        avg_feats = []
        confidences = []
        for i in range(len(curr_anns)):
            ann = curr_anns[i]
            inter_mask = cv2.resize(ann['segmentation'].astype(float), self.predictor.features.shape[-2:], interpolation=cv2.INTER_NEAREST).astype(bool)
            inter_mask = torch.tensor(inter_mask, device=self.predictor.features.device)
            indices = torch.nonzero(inter_mask)
            ann['avg_feat'] = self.predictor.features[0, :, indices[:, 0], indices[:, 1]].mean(dim=1)
            avg_feats.append(ann['avg_feat'])
            confidences.append(ann['predicted_iou'])
        avg_feats_stack = torch.stack(avg_feats)

        # clustering (k-means with k=2)
        cluster_ids, cluster_centers = kmeans(
            X=avg_feats_stack, num_clusters=2, distance='cosine', device=avg_feats_stack.device
        )
        cluster_0_indices = (cluster_ids == 0).nonzero(as_tuple=True)[0]
        cluster_1_indices = (cluster_ids == 1).nonzero(as_tuple=True)[0]
        max_confidence_indices = []
        if len(cluster_0_indices) > 0 and len(cluster_1_indices) > 0:
            index_0 = max(cluster_0_indices, key=lambda i: confidences[i])
            index_1 = max(cluster_1_indices, key=lambda i: confidences[i])
            if curr_anns[index_0]['area'] > curr_anns[index_1]['area']:
                back_index, front_index = index_0, index_1
            else:
                back_index, front_index = index_1, index_0
            final_pred = (curr_anns[front_index]['pred_mask'] + (1-curr_anns[back_index]['pred_mask'])) / 2
            max_confidence_indices.extend([front_index, back_index])
        else:
            if len(cluster_1_indices) == 0:
                index_0 = max(cluster_0_indices, key=lambda i: confidences[i])
                final_pred = curr_anns[index_0]['pred_mask']
                max_confidence_indices.append(index_0)
            elif len(cluster_0_indices) == 0:
                index_1 = max(cluster_1_indices, key=lambda i: confidences[i])
                final_pred = curr_anns[index_1]['pred_mask']
                max_confidence_indices.append(index_1)
            else:
                raise ValueError
        return curr_anns, final_pred, max_confidence_indices
