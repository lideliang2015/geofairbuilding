"""
SegFormer 简化评估脚本（零样本推理）
只评估预训练模型在测试集上的性能，无需训练
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

from config import *
from dataset import get_test_loader
from utils import compute_metrics

# 评估层级
EVALUATION_LEVELS = ['continent', 'hdi_level']


class SegFormerZeroShot(nn.Module):
    """SegFormer 零样本评估包装器"""
    
    def __init__(self, model_name='nvidia/mit-b2', device='cuda'):
        super().__init__()
        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
            use_safetensors=True  # 关键：强制使用 safetensors
        )
        self.model.to(device)
        self.model.eval()
        
    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
            # 上采样到输入尺寸
            logits = nn.functional.interpolate(
                logits, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
        return logits


def evaluate_segformer_zero_shot():
    """SegFormer 零样本评估"""
    print("=" * 60)
    print("SegFormer 零样本评估")
    print("=" * 60)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载测试集
    test_loader, metadata = get_test_loader(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("加载 SegFormer 模型...")
    model = SegFormerZeroShot(device=device)
    print("模型加载完成")
    
    # 评估
    results = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating SegFormer")
        for batch in pbar:
            # 处理 batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    images, masks, metadata = batch
                else:
                    images, masks = batch
                    metadata = None
            else:
                images, masks = batch
                metadata = None
            
            images = images.to(device)
            masks = masks.to(device)
            
            # 推理
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            
            batch_size = len(images)
            
            # 处理 metadata
            if metadata is not None and isinstance(metadata, dict):
                tile_ids = metadata.get('tile_id', ['unknown'] * batch_size)
                continents = metadata.get('continent', ['Unknown'] * batch_size)
                hdi_levels = metadata.get('hdi_level', ['Unknown'] * batch_size)
                resolutions = metadata.get('resolution_m', [0.5] * batch_size)
                sources = metadata.get('source', ['Unknown'] * batch_size)
            else:
                tile_ids = [f'unknown_{i}' for i in range(batch_size)]
                continents = ['Unknown'] * batch_size
                hdi_levels = ['Unknown'] * batch_size
                resolutions = [0.5] * batch_size
                sources = ['Unknown'] * batch_size
            
            for i in range(batch_size):
                pred = preds[i]
                target = targets[i]
                
                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                if target.ndim == 3:
                    target = target.squeeze(0)
                
                metrics = compute_metrics(pred, target)
                
                results.append({
                    'tile_id': tile_ids[i] if i < len(tile_ids) else f'unknown_{i}',
                    'continent': continents[i] if i < len(continents) else 'Unknown',
                    'hdi_level': hdi_levels[i] if i < len(hdi_levels) else 'Unknown',
                    'resolution_m': resolutions[i] if i < len(resolutions) else 0.5,
                    'source': sources[i] if i < len(sources) else 'Unknown',
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'iou': metrics['iou']
                })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, 'segformer_zero_shot_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")
    
    # 打印结果
    print("\n" + "=" * 40)
    print("SegFormer 零样本评估结果")
    print("=" * 40)
    
    print(f"\n整体性能:")
    print(f"  F1: {results_df['f1'].mean():.4f}")
    print(f"  Precision: {results_df['precision'].mean():.4f}")
    print(f"  Recall: {results_df['recall'].mean():.4f}")
    print(f"  IoU: {results_df['iou'].mean():.4f}")
    
    # 按大洲统计
    print("\n按大洲统计:")
    for continent in results_df['continent'].unique():
        subset = results_df[results_df['continent'] == continent]
        print(f"  {continent}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")
    
    # 按 HDI 统计
    print("\n按 HDI 等级统计:")
    for hdi in results_df['hdi_level'].unique():
        subset = results_df[results_df['hdi_level'] == hdi]
        print(f"  {hdi}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")
    
    return results_df


if __name__ == "__main__":
    evaluate_segformer_zero_shot()