import os
from pathlib import Path
import random
random.seed(2024)

img_lists_path = Path("img_lists")
target = "sp_COCO_list.txt"
with open(img_lists_path / target, "r") as f:
    all_list = [t.strip() for t in f.readlines()]
random.shuffle(all_list)
test_list = all_list[:100]
valid_list = all_list[100:200]
train_list = all_list[200:]

output_name = "sp_COCO_tamp"
with open(img_lists_path / f"{output_name}_train.txt", "w") as f:
    f.write('\n'.join(train_list) + '\n')
with open(img_lists_path / f"{output_name}_valid.txt", "w") as f:
    f.write('\n'.join(valid_list) + '\n')
with open(img_lists_path / f"{output_name}_test.txt", "w") as f:
    f.write('\n'.join(test_list) + '\n')
