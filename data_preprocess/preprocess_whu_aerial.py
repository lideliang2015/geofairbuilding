import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd

# ========== 用户配置 ==========
COCO_JSON = r"E:\01.WHU\Aerial imagery dataset\COCO0.2\annotation\train.json"
IMAGE_DIR = r"E:\01.WHU\Aerial imagery dataset\COCO0.2\train"
OUTPUT_DIR = r"E:\processed\01.WHU_aerial"
PATCH_SIZE = 512
STRIDE = 256
MIN_BUILDING_PIXELS = 100   # 最小建筑像素数

# WHU Aerial 数据的地理信息（基督城，新西兰）
# 由于原始数据没有嵌入坐标，手动设置中心点
DEFAULT_LONGITUDE = 172.63   # 基督城经度
DEFAULT_LATITUDE = -43.53    # 基督城纬度
DEFAULT_COUNTRY = "New Zealand"
DEFAULT_CONTINENT = "Oceania"
DEFAULT_HDI_VALUE = 0.937    # 新西兰 HDI 2022
DEFAULT_HDI_LEVEL = "developed"
DEFAULT_RESOLUTION = 0.075   # 米/像素

# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("WHU Aerial 数据集预处理脚本")
print("=" * 60)

# ----- 1. 加载 COCO JSON -----
print("\n[1/3] 加载 COCO 标注文件...")
with open(COCO_JSON, 'r') as f:
    coco = json.load(f)

# 构建映射
images = {img['id']: img for img in coco['images']}
anns_by_img = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    anns_by_img.setdefault(img_id, []).append(ann)

print(f"    - 影像数量: {len(images)}")
print(f"    - 标注数量: {len(coco['annotations'])}")

# ----- 辅助函数 -----
def create_mask(img_info, annotations):
    """从 COCO 多边形标注生成二值掩膜"""
    height = img_info['height']
    width = img_info['width']
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        seg = ann['segmentation']
        if isinstance(seg, list):
            # 多边形格式
            for poly in seg:
                points = np.array(poly).reshape(-1, 2).tolist()
                if len(points) < 3:
                    continue
                img_pil = Image.fromarray(mask)
                draw = ImageDraw.Draw(img_pil)
                draw.polygon(points, outline=1, fill=1)
                mask = np.array(img_pil)
        else:
            # RLE 格式（跳过）
            print(f"    RLE not supported for {img_info['file_name']}")
    return (mask > 0).astype(np.uint8) * 255

def compute_building_density(mask_patch):
    """计算瓦片中建筑像素占比（mask_patch 值为 0 或 255）"""
    return np.mean(mask_patch / 255.0)

def estimate_settlement_type(density):
    """根据建筑密度估算街区类型"""
    if density > 0.3:
        return "urban_high_density"
    elif density > 0.15:
        return "urban_medium_density"
    elif density > 0.05:
        return "suburban"
    elif density > 0.01:
        return "rural"
    else:
        return "sparse"

# ----- 2. 处理影像 -----
print("\n[2/3] 处理影像...")
metadata_records = []

for img_id, img_info in tqdm(images.items(), desc="处理进度"):
    img_filename = img_info['file_name']
    img_path = os.path.join(IMAGE_DIR, img_filename)
    if not os.path.exists(img_path):
        print(f"    警告: 影像不存在 - {img_path}，跳过")
        continue
    
    # 读取原始影像
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    
    # 生成掩膜
    mask = create_mask(img_info, anns_by_img.get(img_id, []))
    
    # 检查掩膜是否有效（是否有建筑）
    if np.sum(mask > 0) == 0:
        continue
    
    # 切块
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            # 裁剪影像和掩膜
            img_patch = img.crop((x, y, x+PATCH_SIZE, y+PATCH_SIZE))
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # 计算建筑密度
            building_density = compute_building_density(mask_patch)
            
            # 跳过建筑占比过低的瓦片
            if np.sum(mask_patch > 0) < MIN_BUILDING_PIXELS:
                continue
            
            # 估算街区类型
            settlement_type = estimate_settlement_type(building_density)
            
            # 保存
            patch_name = f"WHU_Aerial_{img_id}_{x}_{y}.png"
            img_patch.save(os.path.join(OUTPUT_DIR, "images", patch_name))
            mask_patch_img = Image.fromarray(mask_patch)
            mask_patch_img.save(os.path.join(OUTPUT_DIR, "labels", patch_name))
            
            # 记录元数据（与 Inria 脚本字段对齐）
            metadata_records.append({
                "tile_id": patch_name.replace('.png', ''),
                "image_path": f"images/{patch_name}",
                "label_path": f"labels/{patch_name}",
                "source": "WHU_Aerial",
                "original_image": img_filename,
                "resolution_m": DEFAULT_RESOLUTION,
                "width": PATCH_SIZE,
                "height": PATCH_SIZE,
                "x_offset": x,
                "y_offset": y,
                "longitude": DEFAULT_LONGITUDE,
                "latitude": DEFAULT_LATITUDE,
                "country": DEFAULT_COUNTRY,
                "continent": DEFAULT_CONTINENT,
                "hdi_value": DEFAULT_HDI_VALUE,
                "hdi_level": DEFAULT_HDI_LEVEL,
                "building_density": building_density,
                "settlement_type": settlement_type
            })

# ----- 3. 保存元数据 -----
print("\n[3/3] 保存元数据...")
df = pd.DataFrame(metadata_records)
metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
df.to_csv(metadata_path, index=False)

print("=" * 60)
print(f"预处理完成！")
print(f"    - 输出目录: {OUTPUT_DIR}")
print(f"    - 生成瓦片数: {len(metadata_records)}")
print(f"    - 元数据文件: {metadata_path}")

# ----- 统计信息 -----
if len(df) > 0:
    print("\n=== 统计信息 ===")
    print(f"数据源: {df['source'].iloc[0]}")
    print(f"覆盖国家: {df['country'].unique().tolist()}")
    print(f"大洲: {df['continent'].unique().tolist()}")
    print(f"HDI 等级: {df['hdi_level'].unique().tolist()}")
    print(f"分辨率: {df['resolution_m'].unique().tolist()} m")
    print(f"建筑密度范围: {df['building_density'].min():.4f} - {df['building_density'].max():.4f}")
    print(f"街区类型分布:\n{df['settlement_type'].value_counts()}")

print("\n处理完成！")