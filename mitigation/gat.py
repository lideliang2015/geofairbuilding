"""
GAT 实测脚本（适配您的数据格式）
直接加载已训练的 GWL 模型，在验证集上搜索每个大洲的最优阈值
"""

import sys

sys.path.insert(0, 'E:/geofair_experiments')

import os
import torch
import numpy as np
from tqdm import tqdm
from config import *
from dataset import get_dataloaders
from models import get_model


def compute_iou(pred, target):
    """计算IoU"""
    pred_flat = (pred > 0.5).flatten()
    target_flat = (target > 0.5).flatten()
    intersection = np.sum((pred_flat == 1) & (target_flat == 1))
    union = np.sum((pred_flat == 1) | (target_flat == 1))
    return intersection / (union + 1e-6)


def compute_f1(preds, targets):
    """计算F1"""
    preds_flat = (np.array(preds) > 0.5).flatten()
    targets_flat = (np.array(targets) > 0.5).flatten()
    tp = np.sum((preds_flat == 1) & (targets_flat == 1))
    fp = np.sum((preds_flat == 1) & (targets_flat == 0))
    fn = np.sum((preds_flat == 0) & (targets_flat == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def evaluate_gat():
    """GAT 实测流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载 GWL 模型
    model = get_model('segformer', device)
    checkpoint_path = 'E:/experiments/segformer_fair_best.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✓ 加载模型: {checkpoint_path}")

    # 2. 加载验证集和测试集
    _, val_loader, test_loader = get_dataloaders(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
    print(f"✓ 验证集: {len(val_loader.dataset)} 样本")
    print(f"✓ 测试集: {len(test_loader.dataset)} 样本")

    # 3. 在验证集上为每个大洲搜索最优阈值
    print("\n在验证集上搜索最优阈值...")

    # 收集每个大洲的预测概率和真值
    region_probs = {}  # {continent: [probs]}
    region_targets = {}  # {continent: [targets]}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrating"):
            images, masks, metadata = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # 用sigmoid
            continents = metadata['continent']

            for i in range(len(images)):
                region = continents[i]
                if region not in region_probs:
                    region_probs[region] = []
                    region_targets[region] = []
                region_probs[region].append(probs[i])
                region_targets[region].append(masks[i].cpu().numpy())

    # 为每个大洲搜索最优阈值
    thresholds = np.arange(0.3, 0.8, 0.05)
    optimal_thresholds = {}

    for region in region_probs:
        best_f1 = 0
        best_thresh = 0.5

        for thresh in thresholds:
            all_preds = []
            all_targets = []
            for prob, target in zip(region_probs[region], region_targets[region]):
                pred_flat = (prob > thresh).flatten()
                target_flat = (target > 0.5).flatten()
                all_preds.extend(pred_flat)
                all_targets.extend(target_flat)

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            tp = np.sum((all_preds == 1) & (all_targets == 1))
            fp = np.sum((all_preds == 1) & (all_targets == 0))
            fn = np.sum((all_preds == 0) & (all_targets == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds[region] = best_thresh
        print(f"  {region}: optimal threshold = {best_thresh:.2f} (F1={best_f1:.4f})")

    # 4. 在测试集上应用自适应阈值
    print("\n在测试集上应用GAT...")
    region_ious = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, masks, metadata = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            continents = metadata['continent']

            for i in range(len(images)):
                region = continents[i]
                if region not in region_ious:
                    region_ious[region] = []

                thresh = optimal_thresholds.get(region, 0.5)
                pred = (probs[i] > thresh).astype(np.uint8)
                target = masks[i].cpu().numpy()

                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                if target.ndim == 3:
                    target = target.squeeze(0)

                iou = compute_iou(pred, target)
                region_ious[region].append(iou)

    # 5. 计算SFR和Global IoU
    region_performance = {r: np.mean(ious) for r, ious in region_ious.items()}
    iou_values = list(region_performance.values())
    sfr = min(iou_values) / max(iou_values)
    global_iou = np.mean(iou_values)

    print("\n" + "=" * 60)
    print("GAT 评估结果")
    print("=" * 60)
    print(f"SFR: {sfr:.4f}")
    print(f"Global IoU: {global_iou:.4f}")
    print("\n各洲IoU:")
    for region, iou in sorted(region_performance.items()):
        print(f"  {region}: {iou:.4f}")

    # 与GWL对比
    print("\n" + "=" * 60)
    print("与GWL对比")
    print("=" * 60)
    print(f"GWL SFR: 0.6410  |  GWL+GAT SFR: {sfr:.4f}  |  提升: {sfr - 0.641:+.4f}")
    print(f"GWL IoU: 0.5230  |  GWL+GAT IoU: {global_iou:.4f}  |  变化: {global_iou - 0.523:+.4f}")

    return region_performance, sfr, global_iou


if __name__ == "__main__":
    evaluate_gat()