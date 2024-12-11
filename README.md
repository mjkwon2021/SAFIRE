# 💎 SAFIRE

Welcome to the official repository for the paper **"SAFIRE: Segment Any Forged Image Region"**, accepted at AAAI 2025.

SAFIRE specializes in image forgery localization through two methods: **binary localization** and **multi-source partitioning**.  
- **Binary localization** identifies the forged regions in an image  by generating a heatmap that visualizes the probability of each pixel being manipulated.
- **Multi-source partitioning** divides the image into segments based on their originating sources.

---

## 📄 Paper

**Authors**: Myung-Joon Kwon*, Wonjun Lee*, Seung-Hun Nam, Minji Son, and Changick Kim  
**Title**: SAFIRE: Segment Any Forged Image Region  
**Conference**: Proceedings of the AAAI Conference on Artificial Intelligence, 2025  

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

## 📚 Citation

If you find this repository helpful, please consider citing our paper:
```bibtex
@inproceedings{kwon2025safire,
  title={SAFIRE: Segment Any Forged Image Region},
  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
