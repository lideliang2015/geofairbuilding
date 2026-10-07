import os
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ========== 配置 ==========
IMG_DIR = r"E:\01.WHU\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\train\image"
LABEL_DIR = r"E:\01.WHU\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\train\label"
OUTPUT_DIR = r"E:\processed\04.whu_satellite_ii"

# 处理参数
PATCH_SIZE = 512          # 瓦片大小（像素）
STRIDE = 256              # 步长（重叠采样）
MIN_BUILDING_PIXELS = 100 # 最小建筑像素数

# 东亚地区元数据（根据实际情况调整）
CONTINENT = "Asia"
HDI_LEVEL = "developing"  # 东亚多为发展中或发达，统一用 developing
HDI_VALUE = 0.77          # 参考值
RESOLUTION = 1.0          # 约 1m，可根据实际情况调整

# 国家映射（根据文件名或目录推断）
# 如果文件名包含城市信息，可以在这里添加映射
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("WHU Satellite II (East Asia) 预处理脚本")
print("=" * 60)

# ----- 获取所有影像文件 -----
print("\n[1/3] 扫描影像文件...")
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.tif', '.tiff', '.png'))]
print(f"    找到 {len(image_files)} 个影像文件")

if len(image_files) == 0:
    print("错误: 未找到影像文件，请检查路径")
    exit(1)

# ----- 辅助函数 -----
def normalize_image(img):
    """影像归一化到 0-255 uint8"""
    if img.dtype == np.uint16:
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img_norm = np.clip((img - p2) / (p98 - p2) * 255, 0, 255)
        else:
            img_norm = (img / 256).astype(np.float32)
        return img_norm.astype(np.uint8)
    elif img.dtype != np.uint8:
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        return img_norm.astype(np.uint8)
    else:
        return img

def compute_building_density(mask_patch):
    return np.mean(mask_patch / 255.0)

def estimate_settlement_type(density):
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

# ----- 处理每个影像 -----
print("\n[2/3] 开始处理...")
metadata_records = []

for img_file in tqdm(image_files, desc="处理进度"):
    img_path = os.path.join(IMG_DIR, img_file)
    label_file = img_file  # 假设文件名相同
    label_path = os.path.join(LABEL_DIR, label_file)
    
    if not os.path.exists(label_path):
        print(f"    警告: 标签不存在 - {label_file}")
        continue
    
    # 读取影像
    with rasterio.open(img_path) as src:
        img = src.read()
        height, width = src.height, src.width
        # 只保留 RGB 波段（前 3 个）
        if img.shape[0] >= 3:
            img = img[:3, :, :]
        else:
            img = np.stack([img[0], img[0], img[0]], axis=0)
    
    # 读取标签
    with rasterio.open(label_path) as lbl_src:
        lbl = lbl_src.read(1)
    
    # 影像归一化
    img = normalize_image(img)
    
    # 标签二值化（建筑像素 > 0 即为建筑）
    lbl_binary = (lbl > 0).astype(np.uint8) * 255
    
    # 如果影像尺寸小于或等于 PATCH_SIZE，直接使用整图
    if height <= PATCH_SIZE and width <= PATCH_SIZE:
        building_pixels = np.sum(lbl_binary > 0)
        if building_pixels >= MIN_BUILDING_PIXELS:
            building_density = compute_building_density(lbl_binary)
            settlement_type = estimate_settlement_type(building_density)
            
            img_hwc = np.transpose(img, (1, 2, 0))
            img_pil = Image.fromarray(img_hwc)
            lbl_pil = Image.fromarray(lbl_binary)
            
            tile_name = f"WHU_SatII_{os.path.splitext(img_file)[0]}_full.png"
            img_pil.save(os.path.join(OUTPUT_DIR, "images", tile_name))
            lbl_pil.save(os.path.join(OUTPUT_DIR, "labels", tile_name))
            
            metadata_records.append({
                "tile_id": tile_name.replace('.png', ''),
                "image_path": f"images/{tile_name}",
                "label_path": f"labels/{tile_name}",
                "source": "WHU_Satellite_II",
                "original_image": img_file,
                "resolution_m": RESOLUTION,
                "width": width,
                "height": height,
                "x_offset": 0,
                "y_offset": 0,
                "longitude": 0,  # WHU 数据无坐标，可后续手动添加
                "latitude": 0,
                "country": "East Asia",  # 可根据文件名细化
                "continent": CONTINENT,
                "hdi_value": HDI_VALUE,
                "hdi_level": HDI_LEVEL,
                "building_density": building_density,
                "settlement_type": settlement_type
            })
    else:
        # 切块处理
        for y in range(0, height - PATCH_SIZE + 1, STRIDE):
            for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                img_patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                lbl_patch = lbl_binary[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                building_pixels = np.sum(lbl_patch > 0)
                if building_pixels < MIN_BUILDING_PIXELS:
                    continue
                
                building_density = compute_building_density(lbl_patch)
                settlement_type = estimate_settlement_type(building_density)
                
                img_patch_hwc = np.transpose(img_patch, (1, 2, 0))
                img_pil = Image.fromarray(img_patch_hwc)
                lbl_pil = Image.fromarray(lbl_patch)
                
                tile_name = f"WHU_SatII_{os.path.splitext(img_file)[0]}_{x}_{y}.png"
                img_pil.save(os.path.join(OUTPUT_DIR, "images", tile_name))
                lbl_pil.save(os.path.join(OUTPUT_DIR, "labels", tile_name))
                
                metadata_records.append({
                    "tile_id": tile_name.replace('.png', ''),
                    "image_path": f"images/{tile_name}",
                    "label_path": f"labels/{tile_name}",
                    "source": "WHU_Satellite_II",
                    "original_image": img_file,
                    "resolution_m": RESOLUTION,
                    "width": PATCH_SIZE,
                    "height": PATCH_SIZE,
                    "x_offset": x,
                    "y_offset": y,
                    "longitude": 0,
                    "latitude": 0,
                    "country": "East Asia",
                    "continent": CONTINENT,
                    "hdi_value": HDI_VALUE,
                    "hdi_level": HDI_LEVEL,
                    "building_density": building_density,
                    "settlement_type": settlement_type
                })

# ----- 保存元数据 -----
print("\n[3/3] 保存元数据...")
df = pd.DataFrame(metadata_records)
metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
df.to_csv(metadata_path, index=False)

print("=" * 60)
print(f"预处理完成！")
print(f"    - 输出目录: {OUTPUT_DIR}")
print(f"    - 生成瓦片数: {len(metadata_records)}")
print(f"    - 元数据文件: {metadata_path}")

if len(df) > 0:
    print("\n=== 统计信息 ===")
    print(f"数据源: WHU_Satellite_II")
    print(f"大洲: {df['continent'].unique().tolist()}")
    print(f"HDI 等级: {df['hdi_level'].unique().tolist()}")
    print(f"分辨率: {df['resolution_m'].unique().tolist()} m")
    print(f"建筑密度范围: {df['building_density'].min():.4f} - {df['building_density'].max():.4f}")

print("\n处理完成！")