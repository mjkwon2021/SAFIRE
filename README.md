# 💎 SAFIRE

Welcome to the official repository for the paper **"SAFIRE: Segment Any Forged Image Region"**, accepted at AAAI 2025.

SAFIRE specializes in image forgery localization through two methods: **binary localization** and **multi-source partitioning**.  
- **Binary localization** identifies the forged regions in an image  by generating a heatmap that visualizes the probability of each pixel being manipulated.
- **Multi-source partitioning** divides the image into segments based on their originating sources. This task is proposed for the first time in this paper.

---

## 📄 Paper

**Authors**: Myung-Joon Kwon*, Wonjun Lee*, Seung-Hun Nam, Minji Son, and Changick Kim  
**Title**: SAFIRE: Segment Any Forged Image Region  
**Conference**: Proceedings of the AAAI Conference on Artificial Intelligence, 2025  

The paper is available on [[arXiv Link]](https://arxiv.org/abs/2412.08197).

---
## 🎨 Example input / output:

<div style="display: flex; justify-content: space-between; gap: 5px;">
  <figure style="text-align: center; width: 250px;">
    <img src="https://github.com/mjkwon2021/SAFIRE/blob/main/ForensicsEval/inputs/safire_example.png" width="250px">
  </figure>
  
  <figure style="text-align: center; width: 250px;">
    <img src="https://github.com/mjkwon2021/SAFIRE/blob/main/ForensicsEval/outputs_binary/safire_example.png.png" width="250px">
  </figure>
  
  <figure style="text-align: center; width: 250px;">
    <img src="https://github.com/mjkwon2021/SAFIRE/blob/main/ForensicsEval/outputs_multi/safire_example.png.png" width="250px">
  </figure>
</div>

---

## 🎁 SafireMS Dataset

The **SafireMS Dataset** is introduced in our paper and is publicly available on Kaggle for RESEARCH PURPOSES ONLY:  

- **SafireMS-Auto**: Automatically generated datasets used for pretraining.

   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Auto_Splicing-red?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-auto-splicing-dataset)
   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Auto_CopyMove-orange?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-auto-copymove-dataset)
   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Auto_Removal_Part1/2-yellow?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-auto-removal-part-12)
   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Auto_Removal_Part2/2-green?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-auto-removal-part-22)
   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Auto_Generative_Reconstruction-blue?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-auto-generative-reconstruction)

- **SafireMS-Expert**: Manually created datasets designed for evaluating multi-source partitioning performance.  
   [![SafireMS Dataset on Kaggle](https://img.shields.io/badge/Kaggle-SafireMS--Expert-blueviolet?logo=kaggle&style=flat)](https://www.kaggle.com/datasets/qsii24/safire-safirems-expert-multi-source-dataset)

---

## ⚙️ Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/mjkwon2021/SAFIRE.git
   cd SAFIRE
   ```


2. **Download pre-trained weights**  
   Download the weights from [[Google Drive Link]](https://drive.google.com/drive/folders/1NRxep2G42OnVwCR9sGdf1iPqhCUrGmv2).  
   Place the downloaded weights in the root directory of this repository.


3. **Install dependencies**
   ```bash
   conda env create -f environment.yaml
   conda activate safire
   ```
   For manual installation, run the commands listed in `manual_env_setup.txt`.

---

## 🚀 Inference

SAFIRE supports two inference types: **binary forgery localization** and **multi-source partitioning**.

1. **Prepare Input Images**  
   - Place your input images in the directory: `ForensicsEval/inputs`.

2. **Output Locations**  
   - Outputs for binary forgery localization will be saved in: `ForensicsEval/outputs_binary`.  
   - Outputs for multi-source partitioning will be saved in: `ForensicsEval/outputs_multi`.

### Binary Forgery Localization
Run the following command:
```bash
python infer_binary.py --resume="safire.pth"
```

### Multi-Source Partitioning
- **Using k-means clustering**:
  ```bash
  python infer_multi.py --resume="safire.pth" --cluster_type="kmeans" --kmeans_cluster_num=3
  ```
- **Using DBSCAN clustering**:
  ```bash
  python infer_multi.py --resume="safire.pth" --cluster_type="dbscan" --dbscan_eps=0.2 --dbscan_min_samples=1
  ```

---
## 🧪 Test

To evaluate the model on your test dataset:

1. **Download the test dataset**  
   Obtain the test dataset and place it in a desired location.


2. **Set the dataset path**  
   Update the dataset path in `ForensicsEval/project_config.py` to point to your downloaded dataset.


3. **Run the evaluation**  
   - For binary prediction:
     ```bash
     python test_binary.py --resume="safire.pth"
     ```
   - For multi-source partitioning:
     ```bash
     python test_multi.py --resume="safire.pth" --cluster_type="kmeans" --kmeans_cluster_num=3
     ```

4. **View Results**  
   The evaluation results will be saved as an Excel file.

---
## 🏗️ Train

We provide support for distributed data parallel (DDP) training using PyTorch. Below are the instructions to train the model using `train.py`:



Run the following command to start training on multiple GPUs with DDP:

```bash
torchrun --nproc-per-node=6 train.py --batch_size=6 --encresume="safire_encoder_pretrained.pth" --resume="" --num_epochs=150
```

Here are the explanations of the flags:
- `--nproc-per-node`: Specifies the number of GPUs to use on a single node.
- `--batch_size`: Sets the batch size per GPU. In this example, the total batch size is (6 * 6 = 36).
- `--encresume`: Specifies the path to the pretrained encoder checkpoint file. It is uploaded to the Google Drive link provided in the Setup section.
- `--resume`: Specifies the path to the model checkpoint file to resume training. Leave empty (`""`) to start training from scratch.
- `--num_epochs`: Sets the total number of training epochs.

Make sure to adjust these parameters and paths in `ForensicsEval/project_config.py`.


---

## 📚 Citation

If you find this repository helpful, please cite our paper:

[//]: # (AAAI version:)

[//]: # (```bibtex)

[//]: # (@inproceedings{kwon2025safire,)

[//]: # (  title={SAFIRE: Segment Any Forged Image Region},)

[//]: # (  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},)

[//]: # (  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)
[//]: # (arXiv version:)
```bibtex
@article{kwon2024safire,
  title={SAFIRE: Segment Any Forged Image Region},
  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},
  journal={arXiv preprint arXiv:2412.08197},
  year={2024}
}
```
---

## 🔑 Keywords
SAFIRE, Segment Anything Model, SAM, Point Prompting, Promptable Segmentation, Image Forensics, Multimedia Forensics, Image Processing, Image Forgery Detection, Image Forgery Localization, Image Manipulation Detection, Image Manipulation Localization

---
