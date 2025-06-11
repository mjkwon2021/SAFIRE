# -*- coding: utf-8 -*-
"""
Pretrain the encoder.

Myung-Joon Kwon
2024-02-01

Please cite the following paper when using this code:
@article{kwon2024safire,
  title={SAFIRE: Segment Any Forged Image Region},
  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},
  journal={arXiv preprint arXiv:2412.08197},
  year={2024}
}

torchrun --nproc-per-node=6 pretrain.py --batch_size=2
"""

import os

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
import forgery_data_core
import ForensicsEval as FE
import torch.distributed as dist
import gc
from safire_losses import R2R_ContrastiveLoss_TwoSource

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

date_now = datetime.now()
date_now = '%02d%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
IS_MAIN_HOST = GLOBAL_RANK == 0
MAX_FEAT_DICT_SIZE = 1000


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="SAFIRE_pretrain")
parser.add_argument("--model_type", type=str, default="vit_b_adaptor")
parser.add_argument("--checkpoint", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--num_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=8)

# Optimizer parameters
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "--use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")

## Distributed training args
parser.add_argument("--local-rank", type=int, help="local rank")
parser.add_argument(
    "--bucket_cap_mb",
    type=int,
    default=25,
    help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)",
)
parser.add_argument(
    "--grad_acc_steps",
    type=int,
    default=1,
    help="Gradient accumulation steps before syncing gradients for backprop",
)
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--init_method", type=str, default="env://")
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)


def main():
    world_size = WORLD_SIZE
    if IS_MAIN_HOST:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=GLOBAL_RANK, world_size=world_size
    )

    if args.model_type == "vit_b_adaptor":
        sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=args.checkpoint, no_grad_for_backbone=False)
        safire_encoder = sam_model.image_encoder.cuda()
    else:
        raise NotImplementedError

    # train all params
    for param in safire_encoder.parameters():
        param.requires_grad = True

    # mean, std
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()

    safire_encoder = nn.parallel.DistributedDataParallel(
        safire_encoder,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    if IS_MAIN_HOST:
        print(
            "Number of total parameters: ",
            sum(p.numel() for p in safire_encoder.parameters()),
        )
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in safire_encoder.parameters() if p.requires_grad),
        )

    ## Setting up optimiser and loss func
    enc_params = list(
        safire_encoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        enc_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    if IS_MAIN_HOST:
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in enc_params if p.requires_grad),
        )  # 93729252
    pretrain_loss = R2R_ContrastiveLoss_TwoSource()

    # train
    num_epochs = args.num_epochs
    iter_num = 0
    best_valid_epoch_loss = 1e10

    # train dataset
    train_forensic_datasets = [
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_cm_train.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_rm_train.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_sp_train.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_gn_train.txt"),
    ]
    for dataset in train_forensic_datasets:
        dataset.max_num_img = 2000
    train_dataset = forgery_data_core.CoreDataset(train_forensic_datasets, mode="im_mask", augment_type=1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # validation dataset
    valid_forensic_datasets = [
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_cm_valid.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_rm_valid.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_sp_valid.txt"),
        FE.data.Dataset_SafireMSAuto("data/img_lists/SafireMSAuto_gn_valid.txt"),
    ]
    valid_dataset = forgery_data_core.CoreDataset(valid_forensic_datasets, mode="im_mask")
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

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
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(GLOBAL_RANK, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(LOCAL_RANK)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            safire_encoder.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                GLOBAL_RANK,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    for epoch in range(start_epoch, num_epochs):
        gc.collect()
        epoch_loss = 0
        train_dataset.shuffle_im_lists(epoch)
        train_sampler.set_epoch(epoch)
        safire_encoder.train()
        for step, (image, gt2D, img_paths) in enumerate(
                tqdm(train_dataloader, desc=f"[epoch:{epoch}] training...", disable=not IS_MAIN_HOST)
        ):
            if step == 0 and IS_MAIN_HOST:
                print(img_paths[0])
            optimizer.zero_grad()
            image, gt2D = image.cuda(), gt2D.cuda()

            image = (image - pixel_mean) / pixel_std
            pred = safire_encoder(image)
            gt2D = F.interpolate(
                    gt2D.float().unsqueeze(1),
                    size=(pred.shape[2], pred.shape[3]),
                    mode="nearest",
                ).long()

            # Contrastive loss
            loss_cumm = []
            for idx in range(image.shape[0]):
                single_feats = pred[idx].permute(1, 2, 0)
                single_gt2D = gt2D[idx][0]
                query = single_feats[single_gt2D == 0]
                negative = single_feats[single_gt2D == 1]
                if negative.size(0) == 0 or query.size(0) == 0:
                    continue
                query_sample = query[torch.randperm(query.size()[0])[:MAX_FEAT_DICT_SIZE]]
                negative_sample = negative[torch.randperm(negative.size(0))[:MAX_FEAT_DICT_SIZE]]
                loss_cumm.append(pretrain_loss(query_sample, negative_sample))
            loss_cumm = [x for x in loss_cumm if not torch.isnan(x)]
            if len(loss_cumm) > 0:
                loss = torch.mean(torch.stack(loss_cumm).squeeze())
                loss.backward()
                optimizer.step()
            else:
                loss = torch.zeros(1, requires_grad=True).cuda()


            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            epoch_loss += loss.item() * image.shape[0]
            iter_num += 1

        epoch_loss /= len(train_dataset)
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
        safire_encoder.eval()
        with torch.no_grad():
            for step, (image, gt2D, img_paths) in enumerate(
                    tqdm(valid_dataloader, desc=f"[epoch:{epoch}] validating...", disable=not IS_MAIN_HOST)
            ):
                if step == 0 and IS_MAIN_HOST:
                    print(img_paths[0])
                image, gt2D = image.cuda(), gt2D.cuda()

                image = (image - pixel_mean) / pixel_std
                pred = safire_encoder(image)
                gt2D = F.interpolate(
                    gt2D.float().unsqueeze(1),
                    size=(pred.shape[2], pred.shape[3]),
                    mode="nearest",
                ).long()

                # Contrastive loss
                loss_cumm = []
                for idx in range(image.shape[0]):
                    single_feats = pred[idx].permute(1, 2, 0)
                    single_gt2D = gt2D[idx][0]
                    query = single_feats[single_gt2D == 0]
                    negative = single_feats[single_gt2D == 1]
                    if negative.size(0) == 0 or query.size(0) == 0:
                        continue
                    query_sample = query[torch.randperm(query.size()[0])[:MAX_FEAT_DICT_SIZE]]
                    negative_sample = negative[torch.randperm(negative.size(0))[:MAX_FEAT_DICT_SIZE]]
                    loss_cumm.append(pretrain_loss(query_sample, negative_sample))
                loss_cumm = [x for x in loss_cumm if not torch.isnan(x)]
                if len(loss_cumm) > 0:
                    loss = torch.mean(torch.stack(loss_cumm).squeeze())
                else:
                    loss = torch.zeros(1).cuda()

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                valid_epoch_loss += loss.item() * image.shape[0]
        valid_epoch_loss /= len(valid_dataset)

        # save the model checkpoint
        if IS_MAIN_HOST:
            checkpoint = {
                "model": safire_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "pretrained_latest.pth"))

            if (epoch+1) % 50 == 0:
                torch.save(checkpoint, join(model_save_path, f"pretrained_epoch_{epoch}.pth"))

            ## save the best model
            if valid_epoch_loss < best_valid_epoch_loss:
                best_valid_epoch_loss = valid_epoch_loss
                torch.save(checkpoint, join(model_save_path, "pretrained_best.pth"))

        # sync processes
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
