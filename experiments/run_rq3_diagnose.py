"""
RQ3: 地理偏差诊断 - 识别驱动因素
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'E:/geofair_experiments')
from diagnosis.geographical_detector import geographical_detector, geodetector_attribution


def run_rq3_diagnose(output_dir='E:/geofair_experiments/experiments/results'):
    """诊断地理偏差的驱动因素"""
    
    print("=" * 60)
    print("RQ3: Diagnosing Drivers of Geographic Unfairness")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成模拟数据（实际使用时替换为真实数据）
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'iou': np.random.beta(0.5, 5, n_samples),  # 偏斜分布
        'train_density': np.random.exponential(1, n_samples),
        'building_density': np.random.uniform(0, 0.6, n_samples),
        'shape_complexity': np.random.gamma(2, 0.5, n_samples),
        'terrain_roughness': np.random.exponential(0.5, n_samples),
        'cloud_cover': np.random.beta(1, 5, n_samples)
    })
    
    # 添加区域标签
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
    df['continent'] = [continents[i % 6] for i in range(n_samples)]
    
    # 地理探测器分析
    x_cols = ['train_density', 'building_density', 'shape_complexity', 
              'terrain_roughness', 'cloud_cover']
    
    results = geographical_detector(df, 'iou', x_cols, n_strata=5)
    
    print("\n   Geographical Detector Results (q-statistics):")
    print("   " + "-" * 40)
    for driver, stats in results.items():
        print(f"   {driver:20s}: q={stats['q']:.4f}, p={stats['p_value']:.4f}")
    
    # 归因分析
    attribution = geodetector_attribution(df, 'iou', x_cols, n_strata=5)
    
    print("\n   Attribution Analysis:")
    print("   " + "-" * 40)
    for driver, prop in attribution.items():
        print(f"   {driver:20s}: {prop:.2%}")
    
    # 保存结果
    output_file = os.path.join(output_dir, 'rq3_diagnosis_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'geodetector_results': results,
            'attribution': attribution
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "=" * 60)
    print("RQ3 completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_rq3_diagnose()