"""
工具函数
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def compute_metrics(pred, target, threshold=0.5):
    """
    计算单个样本的分割指标
    pred: (H, W) 或 (1, H, W) 预测概率
    target: (H, W) 或 (1, H, W) 真实标签
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    
    # 确保是 2D
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)
    
    # 二值化
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # 展平
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # 计算混淆矩阵
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou': iou
    }

def evaluate_model(model, dataloader, device, metadata_df):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, labels, metas in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            for i in range(len(images)):
                pred = outputs[i]
                target = labels[i]
                if pred.shape[0] == 1:
                    pred = pred.squeeze(0)
                if target.shape[0] == 1:
                    target = target.squeeze(0)
                
                metrics = compute_metrics(pred, target)
                results.append({
                    'tile_id': metas['tile_id'][i],
                    'continent': metas['continent'][i],
                    'hdi_level': metas['hdi_level'][i],
                    'resolution_m': metas['resolution_m'][i],
                    'source': metas['source'][i],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'iou': metrics['iou']
                })
    
    return pd.DataFrame(results)

def compute_fairness_metrics(results_df, group_col='continent'):
    """
    计算公平性指标 RFD 和 FS
    """
    grouped = results_df.groupby(group_col)['f1']
    
    f1_mean = grouped.mean()
    f1_std = grouped.std()
    f1_max = grouped.max()
    f1_min = grouped.min()
    
    # RFD = max - min (单个数值，不是 Series)
    rfd = (f1_max - f1_min).max()  # 确保是标量
    
    # 全局平均 F1
    global_f1 = results_df['f1'].mean()
    
    # FS = 全局平均 / RFD
    fs = global_f1 / rfd if rfd > 0 else float('inf')
    
    return {
        'rfd': rfd,
        'fs': fs,
        'global_f1': global_f1,
        'f1_by_group': f1_mean,
        'std_by_group': f1_std
    }