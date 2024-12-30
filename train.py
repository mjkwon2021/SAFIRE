# -*- coding: utf-8 -*-
"""
Myung-Joon Kwon
2024-02-05

Please cite the following paper when using this code:
@article{kwon2024safire,
  title={SAFIRE: Segment Any Forged Image Region},
  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},
  journal={arXiv preprint arXiv:2412.08197},
  year={2024}
}

Use scheduler
resume from encoder
train the image encoder and mask decoder
freeze prompt image encoder

usage: torchrun --nproc-per-node=6 train.py --batch_size=6 --encresume="safire_encoder_pretrained.pth" --resume="" --num_epochs=150
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
import forgery_data_core
from networks.safire_model import AdaptorSAM
import ForensicsEval as FE
import torch.distributed as dist
import gc
from losses import ILW_BCEWithLogitsLoss, PixelAccWithIgnoreLabel


torch.manual_seed(2024)
torch.cuda.empty_cache()

date_now = datetime.now()
date_now = '%02d%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
IS_MAIN_HOST = GLOBAL_RANK == 0

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="SAFIRE")
parser.add_argument("--model_type", type=str, default="vit_b_adaptor")
parser.add_argument("--checkpoint", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--num_epochs", type=int, default=150)
parser.add_argument("--num_pairs", type=int, default=3)
parser.add_argument("--lambda_ps", type=float, default=0.1)  # pred score loss
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("--bucket_cap_mb", type=int, default=25, help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)",)
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--encresume", type=str, default="", help="Resuming encoder training from checkpoint")
parser.add_argument("--init_method", type=str, default="env://")
args = parser.parse_args()
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)


def main():
    print(f"[Rank {GLOBAL_RANK}]: Use GPU: {LOCAL_RANK} for training")
    if IS_MAIN_HOST:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, os.path.join(model_save_path, os.path.basename(__file__))
        )
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=GLOBAL_RANK, world_size=WORLD_SIZE
    )

    if args.model_type == "vit_b_adaptor":
        sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=args.checkpoint, no_grad_for_backbone=True)  # set False to train all params (default: True)
        safire_model = AdaptorSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).cuda()
    else:
        raise NotImplementedError

    # load encoder weights
    if args.encresume is not None and args.encresume != '':
        if os.path.isfile(args.encresume):
            print(GLOBAL_RANK, "=> loading encoder checkpoint '{}'".format(args.encresume))
            loc = "cuda:{}".format(LOCAL_RANK)
            checkpoint = torch.load(args.encresume, map_location=loc)
            safire_model.image_encoder.load_state_dict({k.replace("module.",""): checkpoint["model"][k] for k in checkpoint["model"]}, strict=False)
            print(
                GLOBAL_RANK,
                "=> loaded encoder checkpoint '{}' (epoch {})".format(
                    args.encresume, checkpoint["epoch"]
                ),
            )
        else:
            raise FileNotFoundError("args.encresume is specified but cannot find the file")

    safire_model = nn.parallel.DistributedDataParallel(
        safire_model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,
        ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    if IS_MAIN_HOST:
        print(
            "Number of total parameters: ",
            sum(p.numel() for p in safire_model.parameters()),
        )  # 93735472
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in safire_model.parameters() if p.requires_grad),
        )  # 93729252
        print(
            "Number of trainable parameters in image_encoder: ",
            sum(p.numel() for p in safire_model.module.image_encoder.parameters() if p.requires_grad),
        )
        print(
            "Number of not trainable parameters in image_encoder: ",
            sum(p.numel() for p in safire_model.module.image_encoder.parameters() if not p.requires_grad),
        )
        print(
            "Number of trainable parameters in mask_decoder: ",
            sum(p.numel() for p in safire_model.module.mask_decoder.parameters() if p.requires_grad),
        )
        print(
            "Number of not trainable parameters in mask_decoder: ",
            sum(p.numel() for p in safire_model.module.mask_decoder.parameters() if not p.requires_grad),
        )

    # only optimize the parameters of image encodder, mask decoder, do not update prompt encoder
    img_mask_encdec_params = list(
        safire_model.module.image_encoder.parameters()
    ) + list(safire_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    if IS_MAIN_HOST:
        print(
            "Number of trainable image encoder and mask decoder parameters: ",
            sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
        )  # 93729252
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    ilw_bce_loss = ILW_BCEWithLogitsLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    acc_with_ignore_label = PixelAccWithIgnoreLabel(ignore_label=-1)

    # train
    num_epochs = args.num_epochs
    iter_num = 0
    train_epoch_losses = []
    best_valid_epoch_loss = 1e10
    best_valid_epoch_loss_bce = 1e10
    best_valid_epoch_loss_ilw_bce = 1e10

    # train dataset
    train_forensic_datasets = [
        FE.data.Dataset_FantasticReality("data/img_lists/FantasticReality_tamp_train.txt"),
        FE.data.Dataset_FantasticReality("data/img_lists/FantasticReality_auth_train.txt"),
        FE.data.Dataset_IMD2020("data/img_lists/IMD2020_tamp_train.txt"),
        FE.data.Dataset_CASIAv2("data/img_lists/CASIA_v2_tamp_train.txt"),
        FE.data.Dataset_CASIAv2("data/img_lists/CASIA_v2_auth_train.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/cm_COCO_tamp_train.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/sp_COCO_tamp_train.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/bcm_COCO_tamp_train.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/bcmc_COCO_tamp_train.txt"),
    ]
    for dataset in train_forensic_datasets:
        dataset.max_num_img = 1869
    train_dataset = forgery_data_core.CoreDataset(train_forensic_datasets, mode="pair_random_cc", num_pairs=args.num_pairs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # validation dataset
    valid_forensic_datasets = [
        FE.data.Dataset_FantasticReality("data/img_lists/FantasticReality_tamp_valid.txt"),
        FE.data.Dataset_FantasticReality("data/img_lists/FantasticReality_auth_valid.txt"),
        FE.data.Dataset_IMD2020("data/img_lists/IMD2020_tamp_valid.txt"),
        FE.data.Dataset_CASIAv2("data/img_lists/CASIA_v2_tamp_valid.txt"),
        FE.data.Dataset_CASIAv2("data/img_lists/CASIA_v2_auth_valid.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/cm_COCO_tamp_valid.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/sp_COCO_tamp_valid.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/bcm_COCO_tamp_valid.txt"),
        FE.data.Dataset_tampCOCO("data/img_lists/bcmc_COCO_tamp_valid.txt"),
    ]
    valid_dataset = forgery_data_core.CoreDataset(valid_forensic_datasets, mode="valid")
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader
    if IS_MAIN_HOST:
        print("Number of training samples: ", len(train_dataset))
        print("Number of validating samples: ", len(valid_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler,
    )

    start_epoch = 0
    if args.resume is not None and args.resume != '':
        if os.path.isfile(args.resume):
            print(GLOBAL_RANK, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(LOCAL_RANK)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            safire_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                GLOBAL_RANK,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        else:
            raise FileNotFoundError("args.resume is specified but cannot find the file")
        torch.distributed.barrier()

    # training loop
    for epoch in range(start_epoch, num_epochs):
        gc.collect()
        epoch_loss = 0
        epoch_ilw_bce_loss = 0
        epoch_pred_score_loss = 0
        train_dataset.shuffle_im_lists(epoch)
        train_sampler.set_epoch(epoch)
        safire_model.train()
        for step, (image, gt2D, points, img_paths) in enumerate(
                tqdm(train_dataloader, desc=f"[epoch:{epoch}] training...", disable=not IS_MAIN_HOST)
        ):
            if step == 0 and IS_MAIN_HOST:
                print(f"First image: {img_paths[0]}")
            optimizer.zero_grad()
            image, gt2D = image.cuda(), gt2D.cuda()
            labels_torch = torch.ones(points.shape[0], points.shape[1]).long()  # (B, 2[auth,tamp])
            point_prompt = (points, labels_torch)

            safire_pred, pred_score = safire_model(image, point_prompt, forward_type=1, upscale_output=False)
            gt2D = F.interpolate(
                    gt2D.float(),
                    size=(safire_pred[0].shape[2], safire_pred[0].shape[3]),
                    mode="nearest",
                ).long()
            basic_losses = [ilw_bce_loss(safire_pred[i], gt2D[:, i:i + 1, ...].float()) for i in range(points.shape[1])]
            basic_loss = sum(basic_losses) / points.shape[1]
            accs = [acc_with_ignore_label(safire_pred[i], gt2D[:, i:i + 1, ...].float()) for i in range(points.shape[1])]
            pred_score_losses = [mse_loss(pred_score[i].squeeze(), accs[i]) for i in range(len(pred_score))]
            pred_score_loss = sum(pred_score_losses) / len(pred_score_losses)
            loss = basic_loss + args.lambda_ps * pred_score_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(basic_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(pred_score_loss, op=dist.ReduceOp.SUM)

            epoch_loss += loss.item() * image.shape[0]
            epoch_ilw_bce_loss += basic_loss.item() * image.shape[0]
            epoch_pred_score_loss += pred_score_loss.item() * image.shape[0]

            iter_num += 1
        epoch_loss /= len(train_dataset)
        epoch_ilw_bce_loss /= len(train_dataset)
        epoch_pred_score_loss /= len(train_dataset)
        train_epoch_losses.append(epoch_loss)
        if IS_MAIN_HOST:
            print(
                f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Epoch Loss: {epoch_loss}'
            )

        # scheduler step
        scheduler.step()

        # sync processes
        torch.distributed.barrier()

        # validation
        valid_epoch_loss = 0
        valid_epoch_loss_bce = 0
        valid_epoch_loss_ilw_bce = 0
        valid_epoch_loss_dice = 0
        safire_model.eval()
        with torch.no_grad():
            for step, (image, gt2D, points, img_paths) in enumerate(
                    tqdm(valid_dataloader, desc=f"[epoch:{epoch}] validating...", disable=not IS_MAIN_HOST)
            ):
                if step == 0 and IS_MAIN_HOST:
                    print(f"First image: {img_paths[0]}")
                image, gt2D = image.cuda(), gt2D.cuda()
                labels_torch = torch.ones(points.shape[0]).long()  # (B,)
                labels_torch = labels_torch.unsqueeze(1)  # (B, 1)
                point_prompt = (points, labels_torch)

                safire_pred, _ = safire_model(image, point_prompt)
                dice_loss_val = dice_loss(safire_pred, gt2D)
                ilw_bce_loss_val = ilw_bce_loss(safire_pred, gt2D)
                bce_loss_val = ce_loss(safire_pred, gt2D.float())
                loss = bce_loss_val + dice_loss_val
                dist.all_reduce(bce_loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(ilw_bce_loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(dice_loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                valid_epoch_loss_bce += bce_loss_val.item() * image.shape[0]
                valid_epoch_loss_ilw_bce += ilw_bce_loss_val.item() * image.shape[0]
                valid_epoch_loss_dice += dice_loss_val.item() * image.shape[0]
                valid_epoch_loss += loss.item() * image.shape[0]
        valid_epoch_loss_bce /= len(valid_dataset)
        valid_epoch_loss_ilw_bce /= len(valid_dataset)
        valid_epoch_loss_dice /= len(valid_dataset)
        valid_epoch_loss /= len(valid_dataset)

        # save the model checkpoint
        if IS_MAIN_HOST:
            checkpoint = {
                "model": safire_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))

            if (epoch+1) % 100 == 0:
                torch.save(checkpoint, os.path.join(model_save_path, f"medsam_model_epoch_{epoch}.pth"))

            ## save the best model
            if valid_epoch_loss < best_valid_epoch_loss:
                best_valid_epoch_loss = valid_epoch_loss
                torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))
            if valid_epoch_loss_bce < best_valid_epoch_loss_bce:
                best_valid_epoch_loss_bce = valid_epoch_loss_bce
                torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best_bce.pth"))
            if valid_epoch_loss_ilw_bce < best_valid_epoch_loss_ilw_bce:
                best_valid_epoch_loss_ilw_bce = valid_epoch_loss_ilw_bce
                torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best_ilw_bce.pth"))

        # %% plot train loss
        if IS_MAIN_HOST:
            plt.plot(train_epoch_losses)
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
            plt.close()

        # sync processes
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
