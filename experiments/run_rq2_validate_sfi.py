"""
RQ2: 验证空间公平性指数 (SFI)
通过合成数据对比 SFR 和 SFI
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, 'E:/geofair_experiments')


def run_rq2_validate_sfi(output_dir='E:/geofair_experiments/experiments/results'):
    """通过合成数据验证 SFI"""
    
    print("=" * 60)
    print("RQ2: Validating Spatial Fairness Index (SFI)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    n_simulations = 1000
    
    results = []
    
    for i in range(n_simulations):
        # 随机生成 SFR 和 Moran's I
        sfr = np.random.uniform(0.3, 1.0)
        morans_i = np.random.uniform(-0.5, 0.9)
        
        # 计算 SFI (w_I=0.5)
        w_I = 0.5
        sfi = sfr * (1 - w_I * max(0, morans_i))
        
        results.append({
            'sim_id': i,
            'sfr': round(sfr, 4),
            'morans_i': round(morans_i, 4),
            'sfi': round(sfi, 4)
        })
    
    df = pd.DataFrame(results)
    
    print("\n   Summary Statistics:")
    print(f"   SFR range: [{df['sfr'].min():.3f}, {df['sfr'].max():.3f}]")
    print(f"   SFI range: [{df['sfi'].min():.3f}, {df['sfi'].max():.3f}]")
    print(f"   Average SFI/SFR ratio: {(df['sfi']/df['sfr']).mean():.3f}")
    
    # 分析不同 Moran's I 下的 SFI 行为
    print("\n   SFI Behavior by Moran's I:")
    print("   " + "-" * 50)
    
    moran_bins = [(-0.5, 0), (0, 0.3), (0.3, 0.6), (0.6, 0.9)]
    for low, high in moran_bins:
        mask = (df['morans_i'] >= low) & (df['morans_i'] < high)
        if mask.any():
            avg_ratio = (df.loc[mask, 'sfi'] / df.loc[mask, 'sfr']).mean()
            print(f"   Moran's I ∈ [{low}, {high}): avg SFI/SFR = {avg_ratio:.3f}")
    
    # 保存结果
    output_file = os.path.join(output_dir, 'rq2_sfi_validation.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_simulations': n_simulations,
            'summary': {
                'sfr_mean': float(df['sfr'].mean()),
                'sfi_mean': float(df['sfi'].mean()),
                'sfi_sfr_ratio_mean': float((df['sfi']/df['sfr']).mean())
            },
            'results': results[:100]
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "=" * 60)
    print("RQ2 completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_rq2_validate_sfi()