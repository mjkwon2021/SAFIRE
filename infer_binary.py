"""
Myung-Joon Kwon
2024-12-10

Infer binary. Uses only one GPU.

Compatible:
networks/safire_predictor_binary.py, networks/safire_model.py

Usage:
python infer_binary.py --resume="safire.pth"
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
from networks.safire_predictor_binary import SafirePredictor
from PIL import Image


torch.cuda.empty_cache()

date_now = datetime.now()
date_now = '%02d%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth")
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

    test_forensic_datasets = {
        "Arbitrary": FE.data.Dataset_Arbitrary(),
    }

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

        with torch.no_grad():
            for step, (image, gt2D, img_paths) in enumerate(
                    tqdm(test_dataloader, desc=f"[Dataset:{dataset_name}] testing...")
            ):
                npimage = (image[0].numpy()).astype(np.uint8)
                anns, safire_pred, max_confidence_indices = safire_automatic_model.safire_predict(npimage)

                save_dir = FE.project_config.dataset_paths["Arbitrary_outputs_binary"]

                # save safire_pred into 0~255 scale binary image to save_dir using PIL
                safire_pred = (safire_pred * 255).astype(np.uint8)
                safire_pred = Image.fromarray(safire_pred)
                safire_pred.save(save_dir / (test_forensic_dataset.get_filename(step) + ".png"))


if __name__ == "__main__":
    main()

