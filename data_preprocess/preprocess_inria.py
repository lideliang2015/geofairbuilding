import os
import numpy as np
import rasterio
from rasterio.warp import transform
from PIL import Image
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ========== 用户配置 ==========
# Inria 数据集路径（请根据您的实际目录结构修改）
INRIA_ROOT = r"E:\02.Inria Aerial Image Labeling\AerialImageDataset"
TRAIN_IMG_DIR = os.path.join(INRIA_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(INRIA_ROOT, "train", "gt")
OUTPUT_DIR = r"E:\processed\02.Inria"

# 处理参数
PATCH_SIZE = 512          # 瓦片大小（像素）
STRIDE = 256              # 步长（重叠采样，增加样本量）
MIN_BUILDING_PIXELS = 100 # 最小建筑像素数，低于此值则跳过该瓦片

# 世界边界 Shapefile 路径（用于从坐标获取国家和大洲）
WORLD_SHAPEFILE = r"E:\data\ne_110m_admin_0_countries.shp"

# HDI Excel 文件路径（用于匹配 HDI 值）
HDI_EXCEL = r"E:\HDI\HDR25_Statistical_Annex_HDI_Table.xlsx"

# 大洲映射（国家名称到大洲的映射，备用）
# 如果世界边界文件包含大洲字段，则无需手动维护
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("Inria 数据集预处理脚本")
print("=" * 60)

# ----- 1. 加载世界边界数据 -----
print("\n[1/4] 加载世界边界数据...")
try:
    world = gpd.read_file(WORLD_SHAPEFILE)
    world = world.to_crs('EPSG:4326')
    # 检查是否有大洲字段（常见字段名：CONTINENT, continent, REGION）
    continent_field = None
    for field in ['CONTINENT', 'continent', 'REGION', 'region']:
        if field in world.columns:
            continent_field = field
            break
    print(f"    - 已加载 {len(world)} 个国家/地区边界")
    print(f"    - 大洲字段: {continent_field if continent_field else '未找到，将手动映射'}")
except Exception as e:
    print(f"    - 错误: {e}")
    world = None
    continent_field = None

# ----- 2. 加载 HDI 数据 -----
print("\n[2/4] 加载 HDI 数据...")
try:
    hdi_df = pd.read_excel(HDI_EXCEL, sheet_name=0)
    # 自动检测列名
    country_col = None
    hdi_col = None
    for col in ['Country', 'country', 'Economy', 'Name', '国家']:
        if col in hdi_df.columns:
            country_col = col
            break
    for col in ['HDI', 'HDI_2023', 'Human Development Index', 'Value', '值']:
        if col in hdi_df.columns:
            hdi_col = col
            break
    
    if country_col is None or hdi_col is None:
        print(f"    - 警告: 无法自动识别列名，请手动修改脚本")
        print(f"      可用列: {hdi_df.columns.tolist()}")
        country_to_hdi = {}
        country_to_level = {}
    else:
        print(f"    - 国家列: '{country_col}'")
        print(f"    - HDI 列: '{hdi_col}'")
        
        def get_hdi_level(val):
            if pd.isna(val):
                return "Unknown"
            if val >= 0.8:
                return "developed"
            elif val >= 0.7:
                return "developing"
            else:
                return "underdeveloped"
        
        country_to_hdi = {}
        country_to_level = {}
        for _, row in hdi_df.iterrows():
            cname = str(row[country_col]).strip().lower()
            hval = row[hdi_col]
            if pd.notna(hval):
                country_to_hdi[cname] = hval
                country_to_level[cname] = get_hdi_level(hval)
        print(f"    - 已加载 {len(country_to_hdi)} 个国家的 HDI 数据")
except Exception as e:
    print(f"    - 错误: {e}")
    country_to_hdi = {}
    country_to_level = {}

# ----- 辅助函数 -----
def get_center_lonlat(tif_path):
    """从 GeoTIFF 获取中心点经纬度（WGS84）"""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        lon, lat = transform(src.crs, 'EPSG:4326', [center_x], [center_y])
        return lon[0], lat[0]

def get_country_and_continent(lon, lat, world_gdf, continent_field):
    """根据经纬度获取国家和大洲"""
    if world_gdf is None:
        return "Unknown", "Unknown"
    point = Point(lon, lat)
    # 空间查询
    country_row = world_gdf[world_gdf.contains(point)]
    if country_row.empty:
        # 缓冲一点再试（防止边界点）
        point_buffered = point.buffer(0.01)
        country_row = world_gdf[world_gdf.intersects(point_buffered)]
    if not country_row.empty:
        country = country_row.iloc[0]['NAME'] if 'NAME' in country_row.columns else country_row.iloc[0]['SOVEREIGNT']
        # 获取大洲
        if continent_field and continent_field in country_row.columns:
            continent = country_row.iloc[0][continent_field]
        else:
            continent = "Unknown"
        return country, continent
    return "Unknown", "Unknown"

def get_hdi_info(country, country_to_hdi, country_to_level):
    """根据国家名称获取 HDI 值和等级"""
    country_lower = country.lower()
    hdi_value = country_to_hdi.get(country_lower, None)
    hdi_level = country_to_level.get(country_lower, "Unknown")
    return hdi_value, hdi_level

def compute_building_density(mask_patch):
    """计算瓦片中建筑像素占比"""
    return np.mean(mask_patch)  # mask_patch 已经是 0/255，除以 255 得 0/1

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

# ----- 3. 获取影像文件列表 -----
print("\n[3/4] 扫描影像文件...")
image_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.tif', '.tiff'))]
if not image_files:
    raise FileNotFoundError(f"未找到影像文件，请检查路径: {TRAIN_IMG_DIR}")
print(f"    - 找到 {len(image_files)} 个影像文件")

# ----- 4. 处理每个影像 -----
print("\n[4/4] 开始处理...")
metadata_records = []

for img_file in tqdm(image_files, desc="处理进度"):
    img_path = os.path.join(TRAIN_IMG_DIR, img_file)
    label_path = os.path.join(TRAIN_LABEL_DIR, img_file)
    
    if not os.path.exists(label_path):
        print(f"    警告: 标签文件不存在 - {label_path}，跳过")
        continue
    
    # ----- 提取坐标和国家信息 -----
    try:
        lon, lat = get_center_lonlat(img_path)
        country, continent = get_country_and_continent(lon, lat, world, continent_field)
        hdi_value, hdi_level = get_hdi_info(country, country_to_hdi, country_to_level)
    except Exception as e:
        print(f"    警告: 坐标提取失败 ({img_file}): {e}")
        lon, lat = 0.0, 0.0
        country, continent = "Unknown", "Unknown"
        hdi_value, hdi_level = None, "Unknown"
    
    # ----- 读取影像和标签 -----
    with rasterio.open(img_path) as src:
        img = src.read()
        height, width = src.height, src.width
        # 只保留 RGB 三个波段
        if img.shape[0] >= 3:
            img = img[:3, :, :]
        else:
            # 单波段复制为 RGB
            img = np.stack([img[0], img[0], img[0]], axis=0)
    
    with rasterio.open(label_path) as lbl_src:
        lbl = lbl_src.read(1)
    
    # ----- 影像归一化到 0-255 -----
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
    
    # 标签二值化（Inria 标签中建筑=255，背景=0）
    lbl_binary = (lbl > 0).astype(np.uint8) * 255
    
    # ----- 切块 -----
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            img_patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            lbl_patch = lbl_binary[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # 计算建筑密度
            building_density = compute_building_density(lbl_patch / 255.0)
            
            # 跳过建筑太少的瓦片
            if np.sum(lbl_patch > 0) < MIN_BUILDING_PIXELS:
                continue
            
            # 估算街区类型
            settlement_type = estimate_settlement_type(building_density)
            
            # 转换并保存
            img_patch_hwc = np.transpose(img_patch, (1, 2, 0))
            img_pil = Image.fromarray(img_patch_hwc)
            lbl_pil = Image.fromarray(lbl_patch)
            
            tile_name = f"Inria_{os.path.splitext(img_file)[0]}_{x}_{y}.png"
            img_pil.save(os.path.join(OUTPUT_DIR, "images", tile_name))
            lbl_pil.save(os.path.join(OUTPUT_DIR, "labels", tile_name))
            
            # 记录元数据
            metadata_records.append({
                "tile_id": tile_name.replace('.png', ''),
                "image_path": f"images/{tile_name}",
                "label_path": f"labels/{tile_name}",
                "source": "Inria",
                "original_image": img_file,
                "resolution_m": 0.3,           # Inria 原始分辨率 0.3m
                "width": PATCH_SIZE,
                "height": PATCH_SIZE,
                "x_offset": x,
                "y_offset": y,
                "longitude": lon,
                "latitude": lat,
                "country": country,
                "continent": continent,
                "hdi_value": hdi_value,
                "hdi_level": hdi_level,
                "building_density": building_density,
                "settlement_type": settlement_type
            })

# ----- 保存元数据 -----
print("\n" + "=" * 60)
df = pd.DataFrame(metadata_records)
metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
df.to_csv(metadata_path, index=False)

print(f"预处理完成！")
print(f"    - 输出目录: {OUTPUT_DIR}")
print(f"    - 生成瓦片数: {len(metadata_records)}")
print(f"    - 元数据文件: {metadata_path}")

# ----- 统计信息 -----
if len(df) > 0:
    print("\n=== 统计信息 ===")
    print(f"覆盖的国家: {df['country'].unique().tolist()}")
    print(f"大洲分布:\n{df['continent'].value_counts()}")
    print(f"HDI 等级分布:\n{df['hdi_level'].value_counts()}")
    print(f"分辨率: {df['resolution_m'].unique().tolist()} m")
    print(f"街区类型分布:\n{df['settlement_type'].value_counts()}")

print("\n处理完成！")