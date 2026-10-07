"""
一键运行所有实验
"""

import subprocess
import sys
import os

from config import OUTPUT_DIR

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"运行: {script_name}")
    print('='*60)
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"错误: {script_name} 执行失败")
        return False
    return True

def main():
    print("=" * 60)
    print("GeoFair-Building v1.0 实验脚本")
    print("=" * 60)
    print("将依次执行:")
    print("  1. U-Net 训练")
    print("  2. SegFormer 训练")
    print("  3. SAM 评估")
    print("  4. 公平性指标计算与对比")
    print("-" * 60)
    
    response = input("是否继续? (y/n): ")
    if response.lower() != 'y':
        return
    
    scripts = ['train_unet.py', 'train_segformer.py', 'evaluate_sam.py', 'evaluate_fairness.py']
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"警告: {script} 不存在，跳过")
            continue
        if not run_script(script):
            print("实验中断")
            break
    
    print(f"\n所有实验完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()