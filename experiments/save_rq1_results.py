"""
保存 RQ1 实验结果
将评估结果保存为 JSON 文件，便于论文使用
"""

import json
import os
from datetime import datetime

# 结果数据
results = {
    "experiment": "RQ1_Geographic_Unfairness",
    "model": "SegFormer (MIT-B2)",
    "timestamp": datetime.now().isoformat(),
    "regional_performance": {
        "Africa": {
            "iou": 0.5172,
            "std": 0.2598,
            "n_samples": 30
        },
        "Asia": {
            "iou": 0.5055,
            "std": 0.1553,
            "n_samples": 36
        },
        "Europe": {
            "iou": 0.6049,
            "std": 0.2440,
            "n_samples": 24
        },
        "North America": {
            "iou": 0.5389,
            "std": 0.2220,
            "n_samples": 24
        },
        "Oceania": {
            "iou": 0.6591,
            "std": 0.1923,
            "n_samples": 12
        },
        "South America": {
            "iou": 0.3817,
            "std": 0.1380,
            "n_samples": 18
        }
    },
    "fairness_metrics": {
        "sfr": 0.5792,
        "min_iou": 0.3817,
        "min_region": "South America",
        "max_iou": 0.6591,
        "max_region": "Oceania",
        "gap": 0.2774
    },
    "global_metrics": {
        "mean_iou": 0.5346,
        "total_samples": 144
    }
}

# 确保目录存在
output_dir = "E:/geofair_experiments/experiments/results"
os.makedirs(output_dir, exist_ok=True)

# 保存 JSON 文件
output_file = os.path.join(output_dir, "rq1_segformer_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ 结果已保存到: {output_file}")
print("\n结果摘要:")
print(f"  空间公平性比率 (SFR): {results['fairness_metrics']['sfr']:.4f}")
print(f"  全局平均 IoU: {results['global_metrics']['mean_iou']:.4f}")
print(f"  性能差距: {results['fairness_metrics']['gap']:.4f}")
print("\n区域性能:")
for region, stats in results['regional_performance'].items():
    print(f"  {region:15s}: IoU = {stats['iou']:.4f} ± {stats['std']:.4f} (n={stats['n_samples']})")