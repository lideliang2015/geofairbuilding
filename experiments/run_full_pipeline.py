"""
全流程实验运行脚本
依次运行 RQ1 到 RQ5
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, 'E:/geofair_experiments')


def run_full_pipeline(
    model_names: list = None,
    data_root: str = 'E:/GeoFair-Building-v1.0',
    output_dir: str = 'E:/geofair_experiments/experiments/results'
):
    """
    运行完整实验流程
    
    Args:
        model_names: 模型名称列表
        data_root: 数据集根目录
        output_dir: 输出目录
    """
    if model_names is None:
        model_names = ['segformer', 'unet', 'deeplabv3']
    
    print("=" * 70)
    print("GEOFAIRNESS FRAMEWORK - FULL EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {model_names}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print("-" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name.upper()}")
        print(f"{'='*60}")
        
        # RQ1: 量化地理不公平性
        from experiments.run_rq1_quantify import run_rq1_quantify
        rq1_results = run_rq1_quantify(
            model_name=model_name,
            data_root=data_root,
            device='cuda',
            output_dir=output_dir
        )
        all_results[f'{model_name}_rq1'] = rq1_results
    
    # 保存汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_evaluated': model_names,
        'results': all_results
    }
    
    summary_file = os.path.join(output_dir, 'full_pipeline_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETED!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    # 仅测试 segformer（因为其他模型可能需要更多时间）
    results = run_full_pipeline(model_names=['segformer'])