"""
公平性指标计算脚本
"""

import torch
import os
import pandas as pd

from config import *
from dataset import get_test_loader
from models import get_model
from utils import evaluate_model, compute_fairness_metrics

def evaluate_trained_model(model_name, model_path=None):
    print("=" * 60)
    print(f"{model_name.upper()} 评估")
    print("=" * 60)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    test_loader, metadata = get_test_loader(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    model = get_model(model_name, device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载权重: {model_path}")
    else:
        print("警告: 未提供权重路径，使用随机初始化模型")
    
    results_df = evaluate_model(model, test_loader, device, metadata)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'{model_name}_results.csv'), index=False)
    
    print("\n" + "=" * 40)
    print(f"{model_name.upper()} 评估结果")
    print("=" * 40)
    print(f"Overall F1: {results_df['f1'].mean():.4f}")
    
    for level in EVALUATION_LEVELS:
        if level in results_df.columns:
            fairness = compute_fairness_metrics(results_df, level)
            print(f"\n【{level}】")
            print(f"  RFD: {fairness['rfd']:.4f}")
            print(f"  FS: {fairness['fs']:.4f}")
    
    return results_df

def compare_models():
    print("=" * 60)
    print("模型对比")
    print("=" * 60)
    
    models = ['unet', 'segformer', 'sam']
    results = {}
    
    for model_name in models:
        result_path = os.path.join(OUTPUT_DIR, f'{model_name}_results.csv')
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            results[model_name] = {'overall_f1': df['f1'].mean(), 'results_df': df}
            print(f"\n{model_name.upper()}: Overall F1 = {df['f1'].mean():.4f}")
    
    print("\n" + "=" * 40)
    print("公平性指标对比")
    print("=" * 40)
    
    for level in EVALUATION_LEVELS:
        print(f"\n【{level}】")
        print(f"{'Model':<12} {'RFD':<10} {'FS':<10} {'Global F1':<10}")
        print("-" * 45)
        
        for model_name in models:
            if model_name in results:
                df = results[model_name]['results_df']
                fairness = compute_fairness_metrics(df, level)
                print(f"{model_name:<12} {fairness['rfd']:<10.4f} {fairness['fs']:<10.4f} {fairness['global_f1']:<10.4f}")

if __name__ == "__main__":
    compare_models()