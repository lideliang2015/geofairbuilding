"""
SAM 零样本评估脚本
评估 Segment Anything Model 在 GeoFair-Building v1.0 测试集上的性能
"""

import torch
import os
import pandas as pd
from tqdm import tqdm

from config import *
from dataset import get_test_loader
from models import get_model
from utils import compute_metrics, compute_fairness_metrics

# 评估层级（根据实际数据字段调整）
EVALUATION_LEVELS = ['continent', 'hdi_level']


def evaluate_model(model, dataloader, device):
    """
    评估模型并返回每个样本的指标
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            # 处理不同格式的 batch
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
            
            # SAM 推理
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            
            batch_size = len(images)
            
            # 处理 metadata
            if metadata is not None:
                if isinstance(metadata, dict):
                    # 字典格式
                    tile_ids = metadata.get('tile_id', ['unknown'] * batch_size)
                    continents = metadata.get('continent', ['Unknown'] * batch_size)
                    hdi_levels = metadata.get('hdi_level', ['Unknown'] * batch_size)
                    resolutions = metadata.get('resolution_m', [0.5] * batch_size)
                    sources = metadata.get('source', ['Unknown'] * batch_size)
                elif isinstance(metadata, (list, tuple)):
                    # 列表格式
                    tile_ids = [m.get('tile_id', f'unknown_{i}') for i, m in enumerate(metadata)]
                    continents = [m.get('continent', 'Unknown') for m in metadata]
                    hdi_levels = [m.get('hdi_level', 'Unknown') for m in metadata]
                    resolutions = [m.get('resolution_m', 0.5) for m in metadata]
                    sources = [m.get('source', 'Unknown') for m in metadata]
                else:
                    tile_ids = [f'unknown_{i}' for i in range(batch_size)]
                    continents = ['Unknown'] * batch_size
                    hdi_levels = ['Unknown'] * batch_size
                    resolutions = [0.5] * batch_size
                    sources = ['Unknown'] * batch_size
            else:
                tile_ids = [f'unknown_{i}' for i in range(batch_size)]
                continents = ['Unknown'] * batch_size
                hdi_levels = ['Unknown'] * batch_size
                resolutions = [0.5] * batch_size
                sources = ['Unknown'] * batch_size
            
            for i in range(batch_size):
                pred = preds[i]
                target = targets[i]
                
                # 确保是 2D
                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                if target.ndim == 3:
                    target = target.squeeze(0)
                
                metrics = compute_metrics(pred, target)
                
                results.append({
                    'tile_id': tile_ids[i],
                    'continent': continents[i],
                    'hdi_level': hdi_levels[i],
                    'resolution_m': resolutions[i],
                    'source': sources[i],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'iou': metrics['iou']
                })
    
    return pd.DataFrame(results)


def evaluate_sam():
    """SAM 评估主函数"""
    print("=" * 60)
    print("SAM 零样本评估")
    print("=" * 60)
    
    # 设备配置
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载测试集
    test_loader, metadata = get_test_loader(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建 SAM 模型
    model = get_model('sam', device)
    print("SAM 模型已加载")
    
    # 评估
    results_df = evaluate_model(model, test_loader, device)
    
    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, 'sam_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")
    
    # 计算并打印整体性能
    print("\n" + "=" * 40)
    print("SAM 评估结果")
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
    
    # 按 HDI 等级统计
    print("\n按 HDI 等级统计:")
    for hdi in results_df['hdi_level'].unique():
        subset = results_df[results_df['hdi_level'] == hdi]
        print(f"  {hdi}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")
    
    # 计算公平性指标
    print("\n" + "=" * 40)
    print("公平性指标")
    print("=" * 40)
    
    for level in EVALUATION_LEVELS:
        if level in results_df.columns:
            fairness = compute_fairness_metrics(results_df, level)
            print(f"\n【{level}】")
            print(f"  RFD: {fairness['rfd']:.4f}")
            print(f"  FS: {fairness['fs']:.4f}")
            print(f"  Global F1: {fairness['global_f1']:.4f}")
            print(f"  F1 by group: {fairness['f1_by_group'].to_dict()}")
    
    return results_df


if __name__ == "__main__":
    evaluate_sam()