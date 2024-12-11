"""
Wonjun Lee, Myung-Joon Kwon
2023-08-22
"""

import numpy as np

import torch
from PIL import Image
import traceback
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

from tqdm import tqdm
from PIL import Image

from sklearn.metrics import confusion_matrix, adjusted_rand_score
from itertools import permutations


def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)
    
    inds = np.argsort(values) 
    inds = inds[(gt0[inds]+gt1[inds])>0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]
        
    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN
    
    msk = np.pad(vet_th[1:]>vet_th[:-1], (0,1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]
    
    return FP, TP, FN, TN, vet_th

def print_unique(gt):
    uv, cnt = np.unique(gt, return_counts=True)
    for v, c in zip(uv, cnt):
        print(f"{v}: {c}")

def computeMetrics_th(values, gt, gt0, gt1, th):    
    values = values>th
    values = values.flatten().astype(np.uint8)
    gt  = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)
    
    gt     = gt[(gt0+gt1)>0]
    values = values[(gt0+gt1)>0]

    cm = confusion_matrix(gt, values, labels=[0, 1])
    
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TP = cm[1, 1]
    
    return FP, TP, FN, TN


def computeMCC(FP, TP, FN, TN):
    FP = np.float64(FP)
    TP = np.float64(TP)
    FN = np.float64(FN)
    TN = np.float64(TN)
    return np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) ), 1e-32)

def computeF1(FP, TP, FN, TN):
    return 2*TP / np.maximum((2*TP + FN + FP), 1e-32)


def calcolaMetriche_threshold(mapp, gt, th):
    FP, TP, FN, TN  = computeMetrics_th(mapp, gt, th)

    f1  = computeF1(FP, TP, FN, TN)
    f1i = computeF1(TN, FN, TP, FP)
    maxF1 = max(f1, f1i)
    
    return 0, maxF1, 0, 0, 0

def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)
    
    # best threshold
    try:
        FP, TP, FN, TN, _  = computeMetricsContinue(map, gt0, gt1)
        f1  = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan
    
    # fixed threshold
    try:
        FP, TP, FN, TN  = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1  = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan
    return F1_best, F1_th
    
    
def computeDetectionMetrics(scores, labels):
    lbl = np.array(labels)
    lbl = lbl[np.isfinite(scores)]
    
    scores = np.array(scores, dtype='float32')
    scores[scores==np.PINF]  = np.nanmax(scores[scores<np.PINF])
    scores = scores[np.isfinite(scores)]
    assert lbl.shape == scores.shape

    # AUC
    from sklearn.metrics import roc_auc_score
    AUC = roc_auc_score(lbl, scores)

    # Balanced Accuracy
    from sklearn.metrics import balanced_accuracy_score
    bACC = balanced_accuracy_score(lbl, scores>0.5)
    
    return AUC, bACC

class test_class_localization():
    def __init__(self, dataset_, is_resize_half = False, is_resize_1024=False, is_auth=True, data_path = 'server18_hard0/mjkwon/forensicsDB', npz_path = "None"):
        """_summary_

        Args:
            dataset_ (_type_): dataset like "Dataset_Carvalho", "Dataset_CASIAv1", etc. only one allowd. not list
            is_resize_half (bool, optional): resize it or not. Defaults to False.
            is_auth (bool, optional): auth dataset(true) or tamp dataset(false). Defaults to True.
            model_name (str, optional): model name. Defaults to "TruFor".
        """
        super().__init__()
        self.dataset = dataset_
        self.is_resize_half = is_resize_half
        self.is_resize_1024 = is_resize_1024
        self.is_auth = is_auth
        self.data_path = data_path
        self.npz_path = npz_path

    def __len__(self):
        return len(self.dataset)
    
    def resize_half(self, img, option=Image.BILINEAR):
        if self.is_resize_half:
            new_w, new_h = img.width//2, img.height//2
            img = img.resize((new_w, new_h), option)
        return img
    
    def resize_1024(self, img, option=Image.BILINEAR):
        if self.is_resize_1024:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
                img = np.array(img.resize((1024, 1024), option))
            else:
                img = img.resize((1024,1024), option)
        return img

    def __getitem__(self, idx):
        img_path = self.dataset.get_img_path(idx)
        # should edit npz_path
        npz_path = str(img_path).replace(self.data_path, self.npz_path)+'.npz'
        npz = np.load(npz_path)
        try:
            mask = self.dataset.get_mask(idx)
            if mask is None:
                mask = np.zeros((1024,1024))
            else:
                mask = self.resize_half(mask, Image.NEAREST)
                mask = self.resize_1024(mask, Image.NEAREST)
        except:
            traceback.print_exc()
            mask = np.nan
        det_gt = 0 if self.is_auth else 1
        gt = (mask, det_gt)
        img_path = str(self.dataset.get_img_path(idx))
        return npz, gt, img_path

class test_class_immediate():
    def __init__(self, dataset_, is_resize_half = False, is_auth=True):
        """_summary_

        Args:
            dataset_ (_type_): dataset like "Dataset_Carvalho", "Dataset_CASIAv1", etc. only one allowd. not list
            is_resize_half (bool, optional): resize it or not. Defaults to False.
            is_auth (bool, optional): auth dataset(true) or tamp dataset(false). Defaults to True.
            model_name (str, optional): model name. Defaults to "TruFor".
        """
        super().__init__()
        self.dataset = dataset_
        self.is_resize_half = is_resize_half
        self.is_auth = is_auth

    def __len__(self):
        return len(self.dataset)
    
    def resize_half(self, img, option=Image.BILINEAR):
        if self.is_resize_half:
            new_w, new_h = img.width//2, img.height//2
            img = img.resize((new_w, new_h), option)
        return img

    def __getitem__(self, idx):
        img_path = str(self.dataset.get_img_path(idx))
        # should edit npz_path
        img = self.dataset.get_img(idx)
        img = self.resize_half(img, option=Image.BILINEAR)
        img = np.array(img.convert("RGB"))
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float)/256.0
        try:
            mask = self.dataset.get_mask(idx)
            mask = self.resize_half(mask, Image.NEAREST)
        except:
            traceback.print_exc()
            mask = np.nan
        det_gt = 0 if self.is_auth else 1
        gt = (mask, det_gt)
        return img, gt, img_path


def f1_best(data_loader):
    f1_best_list = []
    with torch.no_grad():
        for index, (npz, gt, img_path) in enumerate(tqdm(data_loader)):
            print(img_path)
            pred = np.array(npz['pred'])
            gt_mask = np.array(gt[0])
            gt0, gt1 = extractGTs(gt_mask)
            try:
                FP, TP, FN, TN, _ = computeMetricsContinue(pred, gt0, gt1)
                f1 = computeF1(FP, TP, FN, TN)
                f1i = computeF1(TN, FN, TP, FP)
                f1_best = max(np.max(f1), np.max(f1i))
            except:
                traceback.print_exc()
                f1_best = np.nan
            f1_best_list.append(f1_best)
    return np.nanmean(f1_best_list)

def f1_fixed(data_loader):
    f1_fixed_list = []
    with torch.no_grad():
        for index, (npz, gt, img_path) in enumerate(tqdm(data_loader)):
            print(img_path)
            pred = np.array(npz['pred'])
            gt_mask = np.array(gt[0])
            gt0, gt1 = extractGTs(gt_mask)
            try:
                FP, TP, FN, TN = computeMetrics_th(pred, gt_mask, gt0, gt1, 0.5)
                f1 = computeF1(FP, TP, FN, TN)
                f1i = computeF1(TN, FN, TP, FP)
                f1_fixed = max(f1, f1i)
            except:
                traceback.print_exc()
                f1_fixed = np.nan
            f1_fixed_list.append(f1_fixed)
    return np.nanmean(f1_fixed_list)

def mcc(data_loader):
    mcc_list = []
    with torch.no_grad():
        for index, (npz, gt, img_path) in enumerate(tqdm(data_loader)):
            pred = np.array(npz['pred'])
            gt_mask = np.array(gt[0])
            gt0, gt1 = extractGTs(gt_mask)
            try:
                FP, TP, FN, TN = computeMetrics_th(pred, gt_mask, gt0, gt1, 0.5)
                mcc = np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) ), 1e-32)
            except:
                traceback.print_exc()
                mcc = np.nan
            mcc_list.append(mcc)
    return np.nanmean(mcc_list)

def f1_best_tamp(pred, gt_mask):
    gt0, gt1 = extractGTs(gt_mask)
    try:
        FP, TP, FN, TN, _ = computeMetricsContinue(pred, gt0, gt1)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        f1_best = max(np.max(f1), np.max(f1i))
    except:
        traceback.print_exc()
        f1_best = np.nan
    return np.nanmean(f1_best)

def f1_fixed_tamp(pred, gt_mask):
    gt0, gt1 = extractGTs(gt_mask)
    try:
        FP, TP, FN, TN = computeMetrics_th(pred, gt_mask, gt0, gt1, 0.5)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        f1_fixed = max(f1, f1i)
    except:
        traceback.print_exc()
        f1_fixed = np.nan
    return np.nanmean(f1_fixed)

def mcc_tamp(pred, gt_mask):
    gt0, gt1 = extractGTs(gt_mask)
    try:
        FP, TP, FN, TN = computeMetrics_th(pred, gt_mask, gt0, gt1, 0.5)
        mcc = np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP))*np.sqrt((TP + FN))*np.sqrt((TN + FP))*np.sqrt((TN + FN) ), 1e-32)
    except:
        traceback.print_exc()
        mcc = np.nan
    return mcc

def auc(data_loader):
    det_list = []
    gt_det_list = []
    with torch.no_grad():
        for index, (npz, gt, img_path) in enumerate(tqdm(data_loader)):
            det = npz['det']
            gt_det = gt[1]
            det_list.append(det)
            gt_det_list.append(gt_det)
    gt_det_list = np.array(gt_det_list)
    gt_det_list = gt_det_list[np.isfinite(det_list)]

    det_list = np.array(det_list, dtype='float32')
    det_list[det_list==np.PINF]=np.nanmax(det_list[det_list<np.PINF])
    det_list = det_list[np.isfinite(det_list)]
    assert gt_det_list.shape == det_list.shape

    auc = roc_auc_score(gt_det_list, det_list)

    return auc

def pixel_auc(pred, gt_mask):
    pred = pred.flatten()
    gt_mask = gt_mask.flatten()
    assert gt_mask.shape == pred.shape
    auc = roc_auc_score(gt_mask, pred)
    return auc
    
def pixel_AP(pred, gt_mask):
    pred = pred.flatten()
    gt_mask = gt_mask.flatten()
    assert gt_mask.shape == pred.shape
    ap = average_precision_score(gt_mask, pred)
    return ap

def accuracy(data_loader):
    det_list = []
    gt_det_list = []
    with torch.no_grad():
        for index, (npz, gt, img_path) in enumerate(tqdm(data_loader)):
            det = npz['det']
            gt_det = gt[1]
            det_list.append(det)
            gt_det_list.append(gt_det)
    gt_det_list = np.array(gt_det_list)
    gt_det_list = gt_det_list[np.isfinite(det_list)]

    det_list = np.array(det_list, dtype='float32')
    det_list[det_list==np.PINF]=np.nanmax(det_list[det_list<np.PINF])
    det_list = det_list[np.isfinite(det_list)]
    assert gt_det_list.shape == det_list.shape

    acc = balanced_accuracy_score(gt_det_list, det_list>0.5)

    return acc

def det_auc(preds, gts):
    preds = preds.flatten()
    gts = gts.flatten()
    assert gts.shape == preds.shape
    auc = roc_auc_score(gts, preds)
    return auc

def det_balanced_acc(preds, gts):
    preds = preds.flatten()
    gts = gts.flatten()
    assert gts.shape == preds.shape
    acc = balanced_accuracy_score(gts, preds>0.5)
    return acc

def det_AP(preds, gts):
    preds = preds.flatten()
    gts = gts.flatten()
    assert gts.shape == preds.shape
    ap = average_precision_score(gts, preds)
    return ap

def det_mcc(FP, TP, FN, TN):
    mcc = np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP))*np.sqrt((TP + FN))*np.sqrt((TN + FP))*np.sqrt((TN + FN) ), 1e-32)
    return mcc

def st_F1_fixed(FP, TP, FN, TN):
    st_F1_fixed = np.maximum((2 * TP) / np.maximum(1.0, 2 * TP + FN + FP), (2 * FN) / np.maximum(1.0, 2 * FN + TP + TN))
    return st_F1_fixed

def st_acc(FP, TP, FN, TN):
    st_acc = np.maximum((TP + TN) / (TP + TN + FP + FN), (FP + FN) / (TP + TN + FP + FN))
    return st_acc


def normal_miou(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]
    pred_labels = np.zeros_like(pred)
    pred_labels[pred>=0.5] = 1
    miou = mIoU(pred_labels, gt_mask, 2)
    miou_inv = mIoU(1-pred_labels, gt_mask, 2)
    miou_ret = max(miou, miou_inv)
    return miou


def multisource_pmiou(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]

    num_classes = pred.shape[0]

    channel_indices = list(range(num_classes))
    channel_permutations = list(permutations(channel_indices))
    mious = []
    for perm in channel_permutations:
        pred_labels = np.argmax(pred[list(perm), :, :], axis=0)
        miou = mIoU(pred_labels, gt_mask, num_classes)
        mious.append((miou, perm))

    pmiou, perm = max(mious, key=lambda x: x[0])
    return pmiou, perm


def multisource_pmiou_image(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]

    num_classes = np.max(gt_mask)+1

    if pred.shape[0]>num_classes:
        pred_classes = pred.shape[0]
        tmp_pred_labels = np.argmax(pred, axis=0)
        unique, counts = np.unique(tmp_pred_labels, return_counts=True)
        uniq_cnt_dict = dict(zip(unique, counts))

        many_labels = []
        for lbl, cnt in sorted(uniq_cnt_dict.items(), reverse=True, key=lambda x: x[1])[:num_classes]:
            many_labels.append(lbl)

        residual_labels = []
        for lbl, cnt in sorted(uniq_cnt_dict.items(), reverse=True, key=lambda x: x[1])[num_classes:]:
            # print(cnt)
            residual_labels.append(lbl)

        pred_labels_init = np.zeros_like(tmp_pred_labels)
        for i in range(tmp_pred_labels.shape[0]):
            for j in range(tmp_pred_labels.shape[1]):
                for idx, lbl in enumerate(many_labels):
                    if tmp_pred_labels[i][j]==lbl:
                        pred_labels_init[i][j]=idx
                if tmp_pred_labels[i][j] not in many_labels:
                    pred_labels_init[i][j]=-1

        channel_permutations = list(permutations(list(range(num_classes))))

    else:
        many_labels=[]
        channel_permutations = list(permutations(list(range(pred.shape[0]))))
        pred_labels_init = np.argmax(pred, axis=0)

    mious = []
    for perm in channel_permutations:
        pred_labels = pred_labels_init.copy()
        for i in range(pred_labels_init.shape[0]):
            for j in range(pred_labels_init.shape[1]):
                for idx, lbl in enumerate(perm):
                    if pred_labels_init[i][j]==idx:
                        pred_labels[i][j]=lbl
        miou = mIoU(pred_labels, gt_mask, num_classes)

        if many_labels!=[]:
            labels_to_change = [many_labels[i] for i in perm]
            labels_to_change = labels_to_change+residual_labels
        else:
            new_perms = [0] * len(perm)
            for i, p in enumerate(perm):
                new_perms[p] = i
            labels_to_change = new_perms
        mious.append((miou, labels_to_change))

    pmiou, perm = max(mious, key=lambda x: x[0])
    return pmiou, perm

def multisource_pmiou_trunc(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]

    num_classes = np.max(gt_mask)+1

    if pred.shape[0]>num_classes:
        pred_classes = pred.shape[0]
        tmp_pred_labels = np.argmax(pred, axis=0)
        unique, counts = np.unique(tmp_pred_labels, return_counts=True)
        uniq_cnt_dict = dict(zip(unique, counts))

        many_labels = []
        for lbl, cnt in sorted(uniq_cnt_dict.items(), reverse=True, key=lambda x: x[1])[:num_classes]:
            many_labels.append(lbl)

        pred_labels_init = np.zeros_like(tmp_pred_labels)
        for i in range(tmp_pred_labels.shape[0]):
            for j in range(tmp_pred_labels.shape[1]):
                for idx, lbl in enumerate(many_labels):
                    if tmp_pred_labels[i][j]==lbl:
                        pred_labels_init[i][j]=idx
                if tmp_pred_labels[i][j] not in many_labels:
                    pred_labels_init[i][j]=-1

        channel_permutations = list(permutations(list(range(num_classes))))

    else:
        many_labels=[]
        channel_permutations = list(permutations(list(range(pred.shape[0]))))
        pred_labels_init = np.argmax(pred, axis=0)

    mious = []
    for perm in channel_permutations:
        pred_labels = pred_labels_init.copy()
        for i in range(pred_labels_init.shape[0]):
            for j in range(pred_labels_init.shape[1]):
                for idx, lbl in enumerate(perm):
                    if pred_labels_init[i][j]==idx:
                        pred_labels[i][j]=lbl
        miou = mIoU(pred_labels, gt_mask, num_classes)

        if many_labels!=[]:
            labels_to_change = [many_labels[i] for i in perm]
        else:
            labels_to_change = perm
        mious.append((miou, labels_to_change))

    pmiou, perm = max(mious, key=lambda x: x[0])
    return pmiou, perm

def mIoU(pred_labels, gt_labels, num_classes):
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        intersection = np.logical_and(pred_labels == cls, gt_labels == cls).sum()
        union = np.logical_or(pred_labels == cls, gt_labels == cls).sum()
        iou_scores[cls] = intersection / union if union != 0 else 0
    # Calculate mIoU
    miou = np.mean(iou_scores)
    return miou


def normal_ARI(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]
    pred_labels = np.zeros_like(pred)
    pred_labels[pred>=0.5] = 1
    ari = pixel_ARI(pred_labels, gt_mask)

    return ari

def multisource_ARI(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]
    
    pred_labels = np.argmax(pred, axis=0)
    ari = pixel_ARI(pred_labels, gt_mask)

    return ari, None

def multisource_ARI_trunc(pred, gt_mask):
    gt_labels = gt_mask.squeeze()
    assert gt_labels.shape[-2:] == pred.shape[-2:]

    num_classes = np.max(gt_mask)+1
    channel_indices = list(range(num_classes))
    channel_permutations = list(permutations(channel_indices))

    if pred.shape[0]>num_classes:
        pred_classes = pred.shape[0]
        tmp_pred_labels = np.argmax(pred[list(range(pred_classes)), :, :], axis=0)
        unique, counts = np.unique(tmp_pred_labels, return_counts=True)
        uniq_cnt_dict = dict(zip(unique, counts))

        many_labels = []
        for lbl, cnt in sorted(uniq_cnt_dict.items(), reverse=True, key=lambda x: x[1])[:num_classes]:
            many_labels.append(lbl)

        selected_labels = [True if lbl in many_labels else False for lbl in list(range(pred_classes))]
        pred_ = pred[selected_labels, :, :]
    else:
        channel_permutations = list(permutations(list(range(pred.shape[0]))))
        pred_ = pred

    pred_labels = np.argmax(pred_ , axis=0)
    ari = pixel_ARI(pred_labels, gt_mask)

    return ari, None

def pixel_ARI(pred, gt_mask):
    gt_mask, pred = gt_mask.astype(np.int64), pred.astype(np.float32)
    pred = pred.flatten()
    gt_mask = gt_mask.flatten()
    assert gt_mask.shape == pred.shape
    ari = adjusted_rand_score(gt_mask.astype(np.int64), pred.astype(np.float32))
    return ari