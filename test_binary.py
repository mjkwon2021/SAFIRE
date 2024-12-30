"""
Myung-Joon Kwon
2024-12-10

Evaluate binary IFL performance. Uses only one GPU.

Compatible:
networks/safire_predictor_binary.py, networks/safire_model.py

Usage:
python test_binary.py --resume="safire.pth"
"""

import numpy as np
import os

join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import argparse
from datetime import datetime
import forgery_data_core
from networks.safire_model import AdaptorSAM
import ForensicsEval as FE
from pathlib import Path
from ForensicsEval.metric import metrics_functions
from ForensicsEval.fe_utils import AverageMeter
from networks.safire_predictor_binary import SafirePredictor
import easypyxl


torch.cuda.empty_cache()

date_now = datetime.now()
date_now = '%02d%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument("--xlsx", type=str, default="test_results.xlsx", help="xlsx output name")
parser.add_argument("--resume", type=str, default="safire.pth", help="Checkpoint to resume")
parser.add_argument("--points_per_batch", type=int, default=64*4, help="Decrease this if OOM")
parser.add_argument("--points_per_side", type=int, default=16, help="If 16, 16x16 points are used.")
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = Path(os.path.dirname(args.resume))


def main():
    sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=args.sam_checkpoint)
    safire_model = AdaptorSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()

    if args.resume != "":
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            saved_epoch = checkpoint["epoch"]
            safire_model.load_state_dict({k.replace("module.",""): checkpoint["model"][k] for k in checkpoint["model"]})
            print(
                "=> Loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        else:
            raise KeyError(f"Checkpoint file ({args.resume}) not exist!")
    else:
        raise KeyError("Checkpoint file must be given.")

    safire_automatic_model = SafirePredictor(safire_model, points_per_side=args.points_per_side, points_per_batch=args.points_per_batch, pred_iou_thresh=0, stability_score_thresh=0.0, box_nms_thresh=0.0)

    # test datasets
    test_forensic_datasets = {
        "NC16": FE.data.Dataset_NC16("data/img_lists/NC16_tamp.txt"),
        "CocoGlide": FE.data.Dataset_CocoGlide("data/img_lists/CocoGlide_tamp.txt"),
        "Columbia": FE.data.Dataset_Columbia("data/img_lists/Columbia_tamp.txt"),
        "COVERAGE": FE.data.Dataset_COVERAGE("data/img_lists/COVERAGE_tamp.txt"),
        "RealTamper": FE.data.Dataset_RealTamper("data/img_lists/realistic-tampering_tamp.txt"),
    }

    # write results on Excel
    workbook = easypyxl.Workbook(str(save_path / str(args.xlsx)))
    cursor = workbook.new_smart_cursor(sheetname=f"{run_id}", start_cell="B2", corner_name=f"epoch:{str(saved_epoch)}")

    safire_model.eval()
    for dataset_name, test_forensic_dataset in test_forensic_datasets.items():
        test_dataset = forgery_data_core.CoreDataset([test_forensic_dataset], mode="test_auto")
        print(f"[Test] Dataset: {dataset_name}, Number of images: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        # metric
        test_metrics = {
            "F1_fixed": metrics_functions.f1_fixed_tamp,
            "F1_best": metrics_functions.f1_best_tamp,
            "AUC": metrics_functions.pixel_auc,
            "AP": metrics_functions.pixel_AP,
            "mcc": metrics_functions.mcc_tamp,
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

        print(f"dataset:{dataset_name}")
        for metric, result in test_results.items():
            print(f"{metric}: {result.average():04f}")
            cursor.write_cell(dataset_name, metric, result.average())


if __name__ == "__main__":
    main()

