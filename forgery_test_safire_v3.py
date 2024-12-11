"""
Myung-Joon Kwon
2024-01-30

Using only one GPU.

python forgery_test_safire.py --resume="work_dir/ForSAM-Adaptor-20240111-143412/safire.pth" --neptune

"""

# -*- coding: utf-8 -*-

# %% setup environment
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # for windows
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import forgery_data_core
from networks.point_prompt_SAM import MedSAM
from segment_anything.modeling import Sam
from networks.safire_model import AdaptorMedSAM
import ForensicsEval as FE
import neptune
import secret_project_config
from collections import defaultdict
import torch.distributed as dist
import easypyxl
from pathlib import Path

from ForensicsEval.metric import metrics_functions
from ForensicsEval.fe_utils import AverageMeter
from networks.safire_predictor_multi import SafirePredictor
# from networks.safire_predictor import SafirePredictor
import easypyxl

# clustering
from kmeans_pytorch import kmeans

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

date_now = datetime.now()
date_now = '%02d%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_anns_one(anns, ax, index):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = anns
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    ann = sorted_anns[index]
    m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask
    ax.imshow(img)


def show_mask_one(anns, ax, index, max_confidence_indices=None, random_color=False, color=None):
    if len(anns) == 0:
        return
    if max_confidence_indices is None:
        max_confidence_indices = []
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ann = anns[index]
    ax.set_autoscale_on(False)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is None:
        color = np.array([255 / 255, 255 / 255, 255 / 255, 1.0])
    pred = ann["pred_mask"]
    coord = ann["point_coords"]
    h, w = pred.shape[-2:]
    mask_image = pred.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if not index in max_confidence_indices:
        show_points(torch.FloatTensor(coord[0]).unsqueeze(0), torch.LongTensor((1,)), ax, marker='.', marker_size=200)
    else:
        show_points(torch.FloatTensor(coord[0]).unsqueeze(0), torch.LongTensor((1,)), ax, marker='D', color='blue', marker_size=80)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375, marker="*", color='green'):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=color, marker=marker, s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker=marker, s=marker_size, edgecolor='white', linewidth=1.25)



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# %% set up parser
parser = argparse.ArgumentParser()

parser.add_argument("-task_name", type=str, default="ForSAM-eval")
parser.add_argument("-model_type", type=str, default="vit_b_adaptor")
# parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
# parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_h_4b8939.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")


# neptune
parser.add_argument("--neptune", action='store_true', help="for neptune logging")
parser.add_argument("--imsave", action='store_true', help="save prediction and ground truth as images")
parser.add_argument("--tags", type=str, nargs="*", default=[], help="for neptune logging")
parser.add_argument("--memo", type=str, default="", help="for neptune logging")

parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
# model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
save_path = Path(os.path.dirname(args.resume))
print(save_path)


def main():
    if args.neptune:
        run = neptune.init_run(
            project="KAIST-CILAB-AA/ForSAM-eval",
            api_token=secret_project_config.api_token,
            tags=args.tags,
            source_files=["forgery_test_safire.py",]
        )
        run["memo"] = args.memo
        run["args"] = args
        run["run_id"] = run_id
    else:
        run = defaultdict(lambda: list())

    if args.model_type == "vit_b":
        sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
        medsam_model = Sam(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).cuda()
    elif args.model_type == "vit_h":
        sam_model = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
        medsam_model = Sam(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).cuda()
    elif args.model_type == "vit_b_adaptor":
        sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=args.checkpoint)
        medsam_model = AdaptorMedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).cuda()
    else:
        raise NotImplementedError
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    if args.resume != "":
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            saved_epoch = checkpoint["epoch"]
            medsam_model.load_state_dict({k.replace("module.",""): checkpoint["model"][k] for k in checkpoint["model"]})
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        else:
            raise KeyError(f"Checkpoint file ({args.resume}) not exist!")
    else:
        pass
        # raise KeyError("Checkpoint file must be given.")

    safire_automatic_model = SafirePredictor(medsam_model, points_per_side=16, points_per_batch=64*4, pred_iou_thresh=0, stability_score_thresh=0.0, box_nms_thresh=0.0)
    # medsam_automatic_model = SamAutomaticMaskGenerator(medsam_model, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95, box_nms_thresh=0.7)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # test datasets
    test_forensic_datasets = {
        # "CCCRN": FE.data.Dataset_CCCRN("data/img_lists/CCCRN_tamp.txt"),
        "NC16": FE.data.Dataset_NC16("data/img_lists/NC16_tamp.txt"),
        # "NC16_mini": FE.data.Dataset_NC16("data/img_lists/NC16_mini_tamp.txt"),
        # "Carvalho": FE.data.Dataset_Carvalho("data/img_lists/Carvalho_tamp.txt"),
        # "CASIAv1": FE.data.Dataset_CASIAv1("data/img_lists/CASIA_v1_tamp.txt"),
        "CocoGlide": FE.data.Dataset_CocoGlide("data/img_lists/CocoGlide_tamp.txt"),
        "Columbia": FE.data.Dataset_Columbia("data/img_lists/Columbia_tamp.txt"),
        # "CoMoFoD": FE.data.Dataset_CoMoFoD("data/img_lists/CoMoFoD_tamp.txt"),
        "COVERAGE": FE.data.Dataset_COVERAGE("data/img_lists/COVERAGE_tamp.txt"),
        # "GRIP": FE.data.Dataset_GRIP("data/img_lists/GRIP_tamp.txt"),
        # "AutoSplice": FE.data.Dataset_AutoSplice("data/img_lists/AutoSplice_tamp_test.txt"),
        "RealTamper": FE.data.Dataset_RealTamper("data/img_lists/realistic-tampering_tamp.txt"),
        # "val_CASIAv2": FE.data.Dataset_CASIAv2("data/img_lists/CASIA_v2_tamp_valid.txt"),
        # "val_spCOCO": FE.data.Dataset_tampCOCO("data/img_lists/sp_COCO_tamp_valid.txt"),
    }
    # clustering

    # write results on Excel
    workbook = easypyxl.Workbook(str(save_path / "safire_test_16_pred_V3.xlsx"))
    cursor = workbook.new_smart_cursor(sheetname=f"{run_id}", start_cell="B2", corner_name=f"epoch:{str(saved_epoch)}")

    medsam_model.eval()
    for dataset_name, test_forensic_dataset in test_forensic_datasets.items():
        test_dataset = forgery_data_core.CoreDataset([test_forensic_dataset], mode="test_auto")
        print(f"[Test] dataset: {dataset_name}, Number of images: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        # metric
        test_metrics = {
            "F1_fixed": metrics_functions.f1_fixed_tamp,  # pixel level
            "F1_best": metrics_functions.f1_best_tamp,  # pixel level
            "AUC": metrics_functions.pixel_auc,  # pixel level
            "AP": metrics_functions.pixel_AP,  # pixel level
            "mcc": metrics_functions.mcc_tamp,  # pixel level
        }
        test_results = {
            k: AverageMeter() for k in test_metrics
        }
        test_results |= {
            "st_F1_fixed": AverageMeter(),
            "st_Acc": AverageMeter(),
        }
        with torch.no_grad():
            for step, (image, gt2D, img_paths) in enumerate(
                    tqdm(test_dataloader, desc=f"[Dataset:{dataset_name}] testing...")
            ):
                npimage = (image[0].numpy()).astype(np.uint8)
                anns, safire_pred, max_confidence_indices = safire_automatic_model.safire_predict(npimage)

                # Calculate metrics
                pred = safire_pred  # range [0, 1]
                gt = gt2D.numpy()

                for test_metric_name, test_metric in test_metrics.items():
                    result = test_metric(pred, gt)
                    test_results[test_metric_name].update(result)

                pred_r = pred.ravel()
                label_r = gt.squeeze(axis=0).ravel()
                pred_r = pred_r[label_r != -1]
                label_r = label_r[label_r != -1]
                pred_r_binary = (pred_r >= 0.5).astype(int)
                correct = (pred_r_binary == label_r).astype(int)
                incorrect = (pred_r_binary != label_r).astype(int)
                TP = np.count_nonzero(correct[label_r == 1])
                TN = np.count_nonzero(correct[label_r == 0])
                FP = np.count_nonzero(incorrect[label_r == 0])
                FN = np.count_nonzero(incorrect[label_r == 1])

                st_Acc = np.maximum((TP + TN) / (TP + TN + FP + FN), (FP + FN) / (TP + TN + FP + FN))
                test_results["st_Acc"].update(st_Acc)
                st_F1_fixed = np.maximum((2 * TP) / np.maximum(1.0, 2 * TP + FN + FP), (2 * FN) / np.maximum(1.0, 2 * FN + TP + TN))
                test_results["st_F1_fixed"].update(st_F1_fixed)

                # prediction heatmap
                if args.imsave and step % 25 == 0:
                    plt.figure(figsize=(10, 10))
                    plt.title(f"Image: {img_paths[0]}", fontsize=18)
                    plt.axis('off')
                    plt.subplot(2,2,1)
                    plt.axis('off')
                    plt.imshow(image[0].cpu()/255)
                    plt.subplot(2,2,2)
                    plt.axis('off')
                    plt.imshow(gt2D[0].cpu().permute(1, 2, 0), cmap='gray')
                    plt.subplot(2, 2, 3)
                    plt.axis('off')
                    plt.imshow(safire_pred, cmap='gray')
                    plt.subplot(2, 2, 4)
                    plt.axis('off')
                    plt.imshow(np.zeros_like(image[0].cpu()))
                    show_anns(anns, plt.gca())
                    save_dir = save_path / f"pred_{run_id}" / dataset_name
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(fname=str(save_dir / (img_paths[0]+".png")), bbox_inches='tight', pad_inches=0)
                    plt.close()

                # Show all mask (16)
                if args.imsave and step % 25 == 0:
                    plt.figure(figsize=(20, 20))
                    # plt.title(f"Image: {img_paths[0]}, Predicted Score: {iou_pred[0].cpu().item():.3f}, strict F1 fixed: {st_F1_fixed:.3f}", fontsize=18)
                    plt.title(f"Image: {img_paths[0]}", fontsize=18)
                    plt.axis('off')
                    plt.subplot(8,9,1)
                    plt.axis('off')
                    plt.imshow(image[0].cpu()/255)
                    plt.subplot(8,9,2)
                    plt.axis('off')
                    plt.imshow(gt2D[0].cpu().permute(1, 2, 0))
                    for i in range(min(len(anns), 64)):
                        plt.subplot(8,9,3+i)
                        plt.axis('off')
                        plt.title(f"{anns[i]['predicted_iou']:.04f}", fontsize=8)
                        plt.imshow(np.zeros_like(image[0].cpu()))
                        show_mask_one(anns, plt.gca(), i*4, max_confidence_indices)
                    save_dir = save_path / f"pred_{run_id}" / dataset_name
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(fname=str(save_dir / (img_paths[0]+"_masks.png")), bbox_inches='tight', pad_inches=0)
                    plt.close()
        print(f"dataset:{dataset_name}")
        for metric, result in test_results.items():
            print(f"{metric}: {result.average():04f}")
            cursor.write_cell(dataset_name, metric, result.average())


if __name__ == "__main__":
    main()

