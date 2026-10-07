import os
import pandas as pd
import numpy as np
import shutil

# ========== 配置 ==========
PROCESSED_ROOT = r"E:\processed"
OUTPUT_DIR = r"E:\GeoFair-Building-v1.0"
TOTAL_SAMPLES = 960
RANDOM_SEED = 42

# 五个数据集
DATASETS = {
    "WHU_Aerial": {
        "path": os.path.join(PROCESSED_ROOT, "01.WHU_aerial", "metadata.csv"),
        "img_dir": os.path.join(PROCESSED_ROOT, "01.WHU_aerial", "images"),
        "lbl_dir": os.path.join(PROCESSED_ROOT, "01.WHU_aerial", "labels"),
    },
    "Inria": {
        "path": os.path.join(PROCESSED_ROOT, "02.Inria", "metadata.csv"),
        "img_dir": os.path.join(PROCESSED_ROOT, "02.Inria", "images"),
        "lbl_dir": os.path.join(PROCESSED_ROOT, "02.Inria", "labels"),
    },
    "SpaceNet_AOI5": {
        "path": os.path.join(PROCESSED_ROOT, "03.spacenet", "metadata.csv"),
        "img_dir": os.path.join(PROCESSED_ROOT, "03.spacenet", "images"),
        "lbl_dir": os.path.join(PROCESSED_ROOT, "03.spacenet", "labels"),
    },
    "WHU_Satellite_II": {
        "path": os.path.join(PROCESSED_ROOT, "04.whu_satellite_ii", "metadata.csv"),
        "img_dir": os.path.join(PROCESSED_ROOT, "04.whu_satellite_ii", "images"),
        "lbl_dir": os.path.join(PROCESSED_ROOT, "04.whu_satellite_ii", "labels"),
    },
    "SpaceNet_RIO": {
        "path": os.path.join(PROCESSED_ROOT, "05.spacenet_rio", "metadata.csv"),
        "img_dir": os.path.join(PROCESSED_ROOT, "05.spacenet_rio", "images"),
        "lbl_dir": os.path.join(PROCESSED_ROOT, "05.spacenet_rio", "labels"),
    },
}
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("GeoFair-Building v1.0 测试集构建（六大洲完整版）")
print("=" * 60)

# 读取所有元数据
all_dfs = []
for name, config in DATASETS.items():
    if os.path.exists(config["path"]):
        df = pd.read_csv(config["path"])
        df['source'] = name
        print(f"    - {name}: {len(df)} 个瓦片")
        all_dfs.append(df)
    else:
        print(f"    - 警告: {name} 不存在")

merged_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n合并后总瓦片数: {len(merged_df)}")

# 查看分布
print("\n当前数据分布:")
print(f"  大洲:\n{merged_df['continent'].value_counts()}")
print(f"  HDI等级:\n{merged_df['hdi_level'].value_counts()}")

# 论文目标分布
TARGET_BY_CONTINENT = {
    'Asia': 240,
    'Africa': 200,
    'Europe': 160,
    'North America': 160,
    'South America': 120,
    'Oceania': 80,
}

print("\n目标分布（按大洲）:")
for cont, target in TARGET_BY_CONTINENT.items():
    available = len(merged_df[merged_df['continent'] == cont])
    status = "✓" if available >= target else f"⚠ 缺 {target - available}"
    print(f"    {cont}: 目标 {target}, 现有 {available} {status}")

# 分层抽样
selected_indices = []
for continent, target in TARGET_BY_CONTINENT.items():
    continent_df = merged_df[merged_df['continent'] == continent]
    if len(continent_df) == 0:
        print(f"    警告: 没有 {continent} 的数据")
        continue
    
    if len(continent_df) >= target:
        sampled = continent_df.sample(n=target, random_state=RANDOM_SEED)
    else:
        sampled = continent_df
        print(f"    警告: {continent} 只有 {len(continent_df)} 个，全部使用")
    selected_indices.extend(sampled.index.tolist())

final_df = merged_df.loc[selected_indices].reset_index(drop=True)
print(f"\n最终测试集: {len(final_df)} 个瓦片")

# 复制文件
for idx, row in final_df.iterrows():
    source = row['source']
    config = DATASETS.get(source)
    if not config:
        continue
    
    src_img = os.path.join(config["img_dir"], os.path.basename(row['image_path']))
    src_lbl = os.path.join(config["lbl_dir"], os.path.basename(row['label_path']))
    tile_id = row['tile_id']
    
    dst_img = os.path.join(OUTPUT_DIR, "images", f"{tile_id}.png")
    dst_lbl = os.path.join(OUTPUT_DIR, "labels", f"{tile_id}.png")
    
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)

final_df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print("\n=== 最终测试集分布 ===")
print(f"大洲:\n{final_df['continent'].value_counts()}")
print(f"\nHDI等级:\n{final_df['hdi_level'].value_counts()}")
print(f"\n数据源:\n{final_df['source'].value_counts()}")
print(f"\n✅ 完成！输出目录: {OUTPUT_DIR}")