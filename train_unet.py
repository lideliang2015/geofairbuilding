"""
U-Net 训练脚本（优化版）
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import *
from dataset import get_dataloaders
from models import get_model
from utils import dice_loss, compute_metrics


def train_one_epoch(model, loader, optimizer, criterion_bce, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss_bce = criterion_bce(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = CE_WEIGHT * loss_bce + DICE_WEIGHT * loss_dice
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        with torch.no_grad():
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(loader)
    train_metrics = compute_batch_metrics(all_preds, all_targets)
    
    return avg_loss, train_metrics


def validate(model, loader, criterion_bce, device, epoch):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss_bce = criterion_bce(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = CE_WEIGHT * loss_bce + DICE_WEIGHT * loss_dice
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(loader)
    val_metrics = compute_batch_metrics(all_preds, all_targets)
    
    return avg_loss, val_metrics


def compute_batch_metrics(all_preds, all_targets):
    """计算批量指标"""
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
        all_targets = all_targets.squeeze(1)
    
    all_preds_flat = (all_preds > 0.5).flatten().astype(np.uint8)
    all_targets_flat = (all_targets > 0.5).flatten().astype(np.uint8)
    
    tp = np.sum((all_preds_flat == 1) & (all_targets_flat == 1))
    fp = np.sum((all_preds_flat == 1) & (all_targets_flat == 0))
    fn = np.sum((all_preds_flat == 0) & (all_targets_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


def test(model, loader, device):
    """测试集评估"""
    model.eval()
    all_preds = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for images, masks, metadata in pbar:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_metadata.extend(metadata)
    
    results = []
    for i in range(len(all_preds)):
        pred = all_preds[i]
        target = all_targets[i]
        
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        if target.ndim == 3:
            target = target.squeeze(0)
        
        metrics = compute_metrics(pred, target)
        
        # 获取元数据
        if i < len(all_metadata):
            meta = all_metadata[i]
            if isinstance(meta, dict):
                tile_id = meta.get('tile_id', f'unknown_{i}')
                continent = meta.get('continent', 'Unknown')
                hdi_level = meta.get('hdi_level', 'Unknown')
                resolution_m = meta.get('resolution_m', 0.5)
                source = meta.get('source', 'Unknown')
            else:
                tile_id = f'unknown_{i}'
                continent = 'Unknown'
                hdi_level = 'Unknown'
                resolution_m = 0.5
                source = 'Unknown'
        else:
            tile_id = f'unknown_{i}'
            continent = 'Unknown'
            hdi_level = 'Unknown'
            resolution_m = 0.5
            source = 'Unknown'
        
        results.append({
            'tile_id': tile_id,
            'continent': continent,
            'hdi_level': hdi_level,
            'resolution_m': resolution_m,
            'source': source,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'iou': metrics['iou']
        })
    
    return pd.DataFrame(results)


def plot_training_history(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # F1 曲线
    axes[1].plot(history['train_f1'], label='Train F1')
    axes[1].plot(history['val_f1'], label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Training and Validation F1')
    axes[1].legend()
    axes[1].grid(True)
    
    # IoU 曲线
    if 'train_iou' in history and 'val_iou' in history:
        axes[2].plot(history['train_iou'], label='Train IoU')
        axes[2].plot(history['val_iou'], label='Val IoU')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        axes[2].set_title('Training and Validation IoU')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unet_training_history.png'), dpi=150)
    plt.close()


def train_unet():
    """U-Net 训练主函数"""
    print("=" * 60)
    print("U-Net 训练（优化版）")
    print("=" * 60)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载数据
    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_ROOT, BATCH_SIZE, NUM_WORKERS
    )
    print(f"\n训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    model = get_model('unet', device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = None
    if USE_SCHEDULER:
        scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
        print(f"学习率调度: StepLR(step={SCHEDULER_STEP_SIZE}, gamma={SCHEDULER_GAMMA})")
    
    # 损失函数
    criterion_bce = nn.BCEWithLogitsLoss()
    
    # 训练记录
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 
               'train_iou': [], 'val_iou': []}
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_bce, device, epoch
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion_bce, device, epoch
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        
        # 打印
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} [LR: {current_lr:.2e}]")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'unet_best.pth'))
            patience_counter = 0
            print(f"  -> 保存最佳模型 (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  -> 早停于 epoch {epoch+1}")
                break
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        print("-" * 60)
    
    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    # 加载最佳模型
    best_model_path = os.path.join(OUTPUT_DIR, 'unet_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"已加载最佳模型: {best_model_path}")
    
    test_results = test(model, test_loader, device)
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'unet_results.csv'), index=False)
    
    print(f"\n测试集整体性能:")
    print(f"  F1: {test_results['f1'].mean():.4f}")
    print(f"  Precision: {test_results['precision'].mean():.4f}")
    print(f"  Recall: {test_results['recall'].mean():.4f}")
    print(f"  IoU: {test_results['iou'].mean():.4f}")
    
    # 按大洲统计
    print("\n按大洲统计:")
    for continent in test_results['continent'].unique():
        subset = test_results[test_results['continent'] == continent]
        print(f"  {continent}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")
    
    # 按 HDI 统计
    print("\n按 HDI 等级统计:")
    for hdi in test_results['hdi_level'].unique():
        subset = test_results[test_results['hdi_level'] == hdi]
        print(f"  {hdi}: F1 = {subset['f1'].mean():.4f} (n={len(subset)})")
    
    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'unet_history.csv'), index=False)
    
    # 绘制训练曲线
    plot_training_history(history, OUTPUT_DIR)
    
    print(f"\n✅ U-Net 训练完成！")
    print(f"   最佳模型: {os.path.join(OUTPUT_DIR, 'unet_best.pth')}")
    print(f"   测试结果: {os.path.join(OUTPUT_DIR, 'unet_results.csv')}")
    print(f"   训练历史: {os.path.join(OUTPUT_DIR, 'unet_history.csv')}")
    print(f"   训练曲线: {os.path.join(OUTPUT_DIR, 'unet_training_history.png')}")
    
    return model, test_results


if __name__ == "__main__":
    train_unet()