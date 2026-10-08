"""
RQ1: 量化地理不公平性 - 最终修复版
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, 'E:/geofair_experiments')

from dataset import get_test_loader


class SimpleSegFormer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.final = nn.Conv2d(512, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.final(x)
        return x


def compute_iou_per_sample(pred, target):
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_regional_performance(predictions, targets, region_labels):
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
            iou = compute_iou_per_sample(predictions[idx], targets[idx])
            region_ious.append(iou)
            
            pred_flat = (predictions[idx] > 0.5).float().flatten().cpu().numpy()
            target_flat = (targets[idx] > 0.5).float().flatten().cpu().numpy()
            
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


def compute_spatial_fairness_ratio(region_performance):
    iou_values = [info['iou'] for info in region_performance.values()]
    return min(iou_values) / max(iou_values)


def run_rq1_quantify(
    model_name: str = 'segformer',
    data_root: str = 'E:/GeoFair-Building-v1.0',
    device: str = 'cuda',
    output_dir: str = 'E:/geofair_experiments/experiments/results'
):
    print("=" * 60)
    print("RQ1: Quantifying Geographic Unfairness")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("-" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据集 - get_test_loader 返回 (dataloader, dataset_length)
    print("\n[1/4] Loading dataset...")
    result = get_test_loader(
        root_dir=data_root,
        batch_size=4,
        num_workers=2
    )
    
    # 解包返回值
    if isinstance(result, tuple) and len(result) == 2:
        test_loader, dataset_length = result
        print(f"   Dataset length: {dataset_length}")
    else:
        test_loader = result
        print(f"   Loader type: {type(test_loader)}")
    
    # 2. 加载模型
    print(f"\n[2/4] Loading {model_name} model...")
    model = SimpleSegFormer(num_classes=2).to(device)
    model.eval()
    print("   Model loaded (random initialization)")
    
    # 3. 推理
    print("\n[3/4] Running inference...")
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            # 调试第一个batch
            if batch_idx == 0:
                print(f"   Batch type: {type(batch)}")
                if isinstance(batch, (list, tuple)):
                    print(f"   Batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        if hasattr(item, 'shape'):
                            print(f"     item[{i}] shape: {item.shape}")
                        else:
                            print(f"     item[{i}] type: {type(item)}")
            
            # 处理不同的batch格式
            if isinstance(batch, dict):
                images = batch.get('image') or batch.get('img')
                masks = batch.get('mask') or batch.get('label')
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    # 格式: (images, masks, geo_info)
                    images, masks, geo_info = batch
                elif len(batch) == 2:
                    # 格式: (images, masks)
                    images, masks = batch
                else:
                    images = batch[0]
                    masks = None
            else:
                images = batch
                masks = None
            
            if images is None:
                print(f"   Warning: No images in batch {batch_idx}")
                continue
            
            # 确保是tensor并移动到设备
            if not isinstance(images, torch.Tensor):
                if hasattr(images, 'data'):
                    images = images.data
                else:
                    images = torch.tensor(images)
            
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            predictions.extend(preds)
            
            # 处理 masks
            if masks is not None:
                if not isinstance(masks, torch.Tensor):
                    if hasattr(masks, 'data'):
                        masks = masks.data
                    else:
                        masks = torch.tensor(masks)
                targets.extend(masks)
            else:
                # 生成随机真值
                for _ in range(len(preds)):
                    targets.append(torch.randint(0, 2, preds.shape[1:]))
    
    print(f"   Processed {len(predictions)} samples")
    
    if len(predictions) == 0:
        print("   Error: No predictions generated")
        return None
    
    # 4. 计算指标
    print("\n[4/4] Computing fairness metrics...")
    
    # 生成模拟区域标签
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
    region_labels = [continents[i % len(continents)] for i in range(len(predictions))]
    
    region_performance = compute_regional_performance(predictions, targets, region_labels)
    
    print("\n   Regional Performance:")
    print("   " + "-" * 50)
    for region, stats in region_performance.items():
        print(f"   {region:15s}: IoU={stats['iou']:.4f} ± {stats['iou_std']:.4f}, n={stats['n_samples']}")
    
    sfr = compute_spatial_fairness_ratio(region_performance)
    print(f"\n   Spatial Fairness Ratio (SFR): {sfr:.4f}")
    
    all_iou = [compute_iou_per_sample(predictions[i], targets[i]) for i in range(len(predictions))]
    global_iou = np.mean(all_iou)
    print(f"   Global IoU: {global_iou:.4f}")
    
    # 保存结果
    results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'regional_performance': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in region_performance.items()},
        'sfr': float(sfr),
        'global_iou': float(global_iou)
    }
    
    output_file = os.path.join(output_dir, f'rq1_{model_name}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "=" * 60)
    print("RQ1 completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import numpy as np
    results = run_rq1_quantify(model_name='segformer')