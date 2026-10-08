"""
公平性指标计算模块
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def compute_iou_per_sample(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算单样本的IoU"""
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    
    pred = (pred > 0.5).float() if pred.max() > 1 else pred
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()


def compute_regional_performance(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    region_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """计算各区域的性能指标"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    region_results = {}
    unique_regions = set(region_labels)
    
    for region in unique_regions:
        region_indices = [i for i, label in enumerate(region_labels) if label == region]
        
        region_ious = []
        region_precisions = []
        region_recalls = []
        region_f1s = []
        
        for idx in region_indices:
            pred = predictions[idx]
            target = targets[idx]
            
            iou = compute_iou_per_sample(pred, target)
            region_ious.append(iou)
            
            pred_flat = (pred > 0.5).float().flatten().cpu().numpy()
            target_flat = (target > 0.5).float().flatten().cpu().numpy()
            
            region_precisions.append(precision_score(target_flat, pred_flat, zero_division=0))
            region_recalls.append(recall_score(target_flat, pred_flat, zero_division=0))
            region_f1s.append(f1_score(target_flat, pred_flat, zero_division=0))
        
        region_results[region] = {
            'iou': np.mean(region_ious),
            'iou_std': np.std(region_ious),
            'precision': np.mean(region_precisions),
            'recall': np.mean(region_recalls),
            'f1': np.mean(region_f1s),
            'n_samples': len(region_indices)
        }
    
    return region_results


def compute_spatial_fairness_ratio(region_performance: Dict[str, Dict[str, float]]) -> float:
    """计算空间公平性比率 (SFR)"""
    iou_values = [info['iou'] for info in region_performance.values()]
    return min(iou_values) / max(iou_values)


def compute_spatial_fairness_index(
    iou_values: np.ndarray,
    coordinates: np.ndarray,
    sfr: float,
    w_I: float = 0.5,
    threshold_km: float = 50.0
) -> float:
    """计算空间公平性指数 (SFI)"""
    try:
        from .spatial_stats import compute_global_moran_i
        moran_i, _ = compute_global_moran_i(iou_values, coordinates, threshold_km)
        I_P_positive = max(0, moran_i)
        sfi = sfr * (1 - w_I * I_P_positive)
    except ImportError:
        sfi = sfr
    
    return sfi


def compute_equalized_odds_difference(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    region_labels: List[str]
) -> float:
    """计算机会均等差异 (EOD)"""
    unique_regions = set(region_labels)
    tpr_list = []
    fpr_list = []
    
    for region in unique_regions:
        region_indices = [i for i, label in enumerate(region_labels) if label == region]
        
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        
        for idx in region_indices:
            pred = (predictions[idx] > 0.5).float().flatten().cpu().numpy()
            target = (targets[idx] > 0.5).float().flatten().cpu().numpy()
            
            tp += np.sum((pred == 1) & (target == 1))
            fn += np.sum((pred == 0) & (target == 1))
            fp += np.sum((pred == 1) & (target == 0))
            tn += np.sum((pred == 0) & (target == 0))
        
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    eod = max(max(tpr_list) - min(tpr_list), max(fpr_list) - min(fpr_list))
    
    return eod


def compute_hotspot_ratio(
    iou_values: np.ndarray,
    coordinates: np.ndarray,
    p_threshold: float = 0.05,
    threshold_km: float = 50.0
) -> float:
    """计算不公平热点比例"""
    try:
        from .spatial_stats import compute_getis_ord_gi_star
        _, z_scores = compute_getis_ord_gi_star(iou_values, coordinates, threshold_km)
        threshold = 1.96
        hotspots = z_scores < -threshold
        return np.sum(hotspots) / len(hotspots)
    except ImportError:
        threshold = np.percentile(iou_values, 10)
        return np.sum(iou_values < threshold) / len(iou_values)