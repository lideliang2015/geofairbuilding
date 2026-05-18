"""
重新评估 U-Net 并保存正确元数据
"""

import torch
import pandas as pd
from tqdm import tqdm
import os
from config import *
from dataset import get_test_loader
from models import get_model
from utils import compute_metrics

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载测试集
test_loader, metadata = get_test_loader(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
print(f"测试集: {len(test_loader.dataset)} 样本")

# 加载模型
model = get_model('unet', device)
best_model_path = os.path.join(OUTPUT_DIR, 'unet_best.pth')
model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"已加载模型: {best_model_path}")

# 评估
results = []
model.eval()
with torch.no_grad():
    for images, masks, metas in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = torch.sigmoid(logits).cpu().numpy()
        
        batch_size = len(images)
        
        # 处理元数据
        if isinstance(metas, dict):
            tile_ids = metas.get('tile_id', ['unknown'] * batch_size)
            continents = metas.get('continent', ['Unknown'] * batch_size)
            hdi_levels = metas.get('hdi_level', ['Unknown'] * batch_size)
            sources = metas.get('source', ['Unknown'] * batch_size)
        else:
            tile_ids = [f'unknown_{i}' for i in range(batch_size)]
            continents = ['Unknown'] * batch_size
            hdi_levels = ['Unknown'] * batch_size
            sources = ['Unknown'] * batch_size
        
        for i in range(batch_size):
            # 计算指标
            pred = preds[i]
            target = masks[i].cpu().numpy()
            if pred.ndim == 3:
                pred = pred.squeeze(0)
            if target.ndim == 3:
                target = target.squeeze(0)
            metrics = compute_metrics(pred, target)
            
            results.append({
                'tile_id': tile_ids[i],
                'continent': continents[i],
                'hdi_level': hdi_levels[i],
                'source': sources[i],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iou': metrics['iou']
            })

# 保存结果
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, 'unet_results_fixed.csv'), index=False)

# 打印统计
print("\n=== U-Net 评估结果 ===")
print(f"整体 F1: {df['f1'].mean():.4f}")

print("\n按大洲统计:")
for cont in df['continent'].unique():
    subset = df[df['continent'] == cont]
    print(f"  {cont}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")

print("\n按 HDI 统计:")
for hdi in df['hdi_level'].unique():
    subset = df[df['hdi_level'] == hdi]
    print(f"  {hdi}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")