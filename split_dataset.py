"""
数据集划分脚本
将 GeoFair-Building v1.0 划分为训练集、验证集、测试集
划分比例: 70% 训练 / 15% 验证 / 15% 测试
采用分层抽样，确保各大洲和 HDI 等级分布一致
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ========== 配置 ==========
DATASET_ROOT = r"E:\GeoFair-Building-v1.0"
METADATA_PATH = os.path.join(DATASET_ROOT, "metadata.csv")
OUTPUT_PATH = os.path.join(DATASET_ROOT, "splits.csv")
RANDOM_SEED = 42

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 分层字段（按大洲分层，确保地理分布一致）
STRATIFY_COLUMN = 'continent'
# ============================

def main():
    print("=" * 60)
    print("GeoFair-Building v1.0 数据集划分")
    print("=" * 60)
    
    # ----- 1. 读取元数据 -----
    print(f"\n[1/5] 读取元数据: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    print(f"    总样本数: {len(df)}")
    print(f"    字段: {df.columns.tolist()}")
    
    # ----- 2. 查看原始分布 -----
    print("\n[2/5] 原始数据分布:")
    print("\n  大洲分布:")
    print(df['continent'].value_counts())
    print("\n  HDI 等级分布:")
    print(df['hdi_level'].value_counts())
    print("\n  数据源分布:")
    print(df['source'].value_counts())
    
    # ----- 3. 划分数据集 -----
    print("\n[3/5] 执行分层抽样划分...")
    print(f"    划分比例: 训练 {TRAIN_RATIO*100:.0f}% / 验证 {VAL_RATIO*100:.0f}% / 测试 {TEST_RATIO*100:.0f}%")
    print(f"    分层字段: {STRATIFY_COLUMN}")
    print(f"    随机种子: {RANDOM_SEED}")
    
    indices = df.index.tolist()
    
    # 第一次划分：分出测试集
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df[STRATIFY_COLUMN]
    )
    
    # 第二次划分：从训练+验证中分出验证集
    # 验证集比例 = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=df.loc[train_val_idx, STRATIFY_COLUMN]
    )
    
    # ----- 4. 保存划分结果 -----
    print("\n[4/5] 保存划分结果...")
    
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'
    
    # 保存到文件
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"    已保存至: {OUTPUT_PATH}")
    
    # ----- 5. 输出统计信息 -----
    print("\n[5/5] 划分结果统计:")
    print("\n  总体划分:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        pct = count / len(df) * 100
        print(f"    {split}: {count} ({pct:.1f}%)")
    
    print("\n  各数据源划分:")
    print(pd.crosstab(df['source'], df['split']))
    
    print("\n  各大洲划分:")
    continent_cross = pd.crosstab(df['continent'], df['split'])
    print(continent_cross)
    
    print("\n  各大洲在测试集中的比例:")
    test_continent = df[df['split'] == 'test']['continent'].value_counts()
    test_continent_pct = test_continent / len(df[df['split'] == 'test']) * 100
    for cont, count in test_continent.items():
        print(f"    {cont}: {count} ({test_continent_pct[cont]:.1f}%)")
    
    print("\n  HDI 等级划分:")
    print(pd.crosstab(df['hdi_level'], df['split']))
    
    # ----- 验证分层效果 -----
    print("\n" + "=" * 60)
    print("分层效果验证")
    print("=" * 60)
    
    original_pct = df['continent'].value_counts() / len(df) * 100
    test_pct = df[df['split'] == 'test']['continent'].value_counts() / len(df[df['split'] == 'test']) * 100
    
    print("\n  各大洲比例对比:")
    print(f"{'大洲':<15} {'原始比例':<12} {'测试集比例':<12} {'差异':<10}")
    print("-" * 50)
    for cont in original_pct.index:
        orig = original_pct[cont]
        test = test_pct.get(cont, 0)
        diff = abs(orig - test)
        print(f"{cont:<15} {orig:.1f}%{'':<6} {test:.1f}%{'':<6} {diff:.1f}%")
    
    # ----- 保存划分索引文件（用于模型训练）-----
    print("\n" + "=" * 60)
    print("保存划分索引文件")
    print("=" * 60)
    
    train_ids = df[df['split'] == 'train']['tile_id'].tolist()
    val_ids = df[df['split'] == 'val']['tile_id'].tolist()
    test_ids = df[df['split'] == 'test']['tile_id'].tolist()
    
    # 保存为文本文件
    with open(os.path.join(DATASET_ROOT, "train_ids.txt"), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(DATASET_ROOT, "val_ids.txt"), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(DATASET_ROOT, "test_ids.txt"), 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"    训练集 ID: {len(train_ids)} 个 → train_ids.txt")
    print(f"    验证集 ID: {len(val_ids)} 个 → val_ids.txt")
    print(f"    测试集 ID: {len(test_ids)} 个 → test_ids.txt")
    
    print("\n" + "=" * 60)
    print("✅ 数据划分完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()