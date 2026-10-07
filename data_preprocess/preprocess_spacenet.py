import os
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio import features
from PIL import Image
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ========== 用户配置 ==========
AOI5_IMG_DIR = r"E:\03.spacenet\AOI_5_Khartoum\AOI_5_Khartoum_Train\RGB-PanSharpen"
AOI5_LABEL_DIR = r"E:\03.spacenet\AOI_5_Khartoum\AOI_5_Khartoum_Train\geojson\buildings"
OUTPUT_DIR = r"E:\processed\03.spacenet"

PATCH_SIZE = 512
STRIDE = 256
MIN_BUILDING_PIXELS = 100
TARGET_RESOLUTION = 0.5

WORLD_SHAPEFILE = r"E:\data\ne_110m_admin_0_countries.shp"
HDI_EXCEL = r"E:\HDI\HDR25_Statistical_Annex_HDI_Table.xlsx"
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("SpaceNet AOI_5_Khartoum 预处理脚本")
print("=" * 60)

# ----- 1. 加载世界边界数据 -----
print("\n[1/4] 加载世界边界数据...")
world = None
continent_field = None
try:
    world = gpd.read_file(WORLD_SHAPEFILE)
    world = world.to_crs('EPSG:4326')
    for field in ['CONTINENT', 'continent', 'REGION', 'region']:
        if field in world.columns:
            continent_field = field
            break
    print(f"    - 已加载 {len(world)} 个国家/地区边界")
except Exception as e:
    print(f"    - 警告: 无法加载世界边界: {e}")

# ----- 2. 加载 HDI 数据 -----
print("\n[2/4] 加载 HDI 数据...")
country_to_hdi = {}
country_to_level = {}
try:
    hdi_df = pd.read_excel(HDI_EXCEL, sheet_name=0)
    country_col = None
    hdi_col = None
    for col in ['Country', 'country', 'Economy', 'Name']:
        if col in hdi_df.columns:
            country_col = col
            break
    for col in ['HDI', 'HDI_2023', 'Human Development Index', 'Value']:
        if col in hdi_df.columns:
            hdi_col = col
            break
    
    if country_col and hdi_col:
        def get_hdi_level(val):
            if pd.isna(val):
                return "Unknown"
            return "developed" if val >= 0.8 else "developing" if val >= 0.7 else "underdeveloped"
        
        for _, row in hdi_df.iterrows():
            cname = str(row[country_col]).strip().lower()
            hval = row[hdi_col]
            if pd.notna(hval):
                country_to_hdi[cname] = hval
                country_to_level[cname] = get_hdi_level(hval)
        print(f"    - 已加载 {len(country_to_hdi)} 个国家的 HDI 数据")
    else:
        print(f"    - 警告: 无法识别列名")
except Exception as e:
    print(f"    - 警告: 无法加载 HDI 文件: {e}")

# ----- 辅助函数 -----
def get_center_lonlat(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        lon, lat = transform(src.crs, 'EPSG:4326', [center_x], [center_y])
        return lon[0], lat[0]

def get_country_and_continent(lon, lat):
    if world is None:
        return "Sudan", "Africa"
    point = Point(lon, lat)
    country_row = world[world.contains(point)]
    if country_row.empty:
        point_buffered = point.buffer(0.01)
        country_row = world[world.intersects(point_buffered)]
    if not country_row.empty:
        country = country_row.iloc[0].get('NAME', country_row.iloc[0].get('SOVEREIGNT', 'Unknown'))
        if continent_field and continent_field in country_row.columns:
            continent = country_row.iloc[0][continent_field]
        else:
            continent = "Unknown"
        return country, continent
    return "Sudan", "Africa"

def get_hdi_info(country):
    country_lower = country.lower()
    hdi_value = country_to_hdi.get(country_lower, None)
    hdi_level = country_to_level.get(country_lower, "Unknown")
    return hdi_value, hdi_level

def rasterize_geojson(geojson_path, transform, shape):
    try:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            return np.zeros(shape, dtype=np.uint8)
        shapes = ((geom, 1) for geom in gdf.geometry)
        mask = features.rasterize(shapes, out_shape=shape, transform=transform, fill=0)
        return (mask > 0).astype(np.uint8) * 255
    except Exception as e:
        return np.zeros(shape, dtype=np.uint8)

def normalize_image(img):
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

# ----- 3. 处理 AOI_5 -----
print("\n[3/4] 开始处理 AOI_5_Khartoum...")

if not os.path.exists(AOI5_IMG_DIR):
    raise FileNotFoundError(f"影像目录不存在: {AOI5_IMG_DIR}")
if not os.path.exists(AOI5_LABEL_DIR):
    raise FileNotFoundError(f"标签目录不存在: {AOI5_LABEL_DIR}")

# 获取所有影像文件
image_files = [f for f in os.listdir(AOI5_IMG_DIR) if f.endswith(('.tif', '.tiff'))]
print(f"    找到 {len(image_files)} 个影像文件")

metadata_records = []

for img_file in tqdm(image_files, desc="处理进度"):
    img_path = os.path.join(AOI5_IMG_DIR, img_file)
    
    # ===== 修正：构建正确的标签文件名 =====
    # 影像: RGB-PanSharpen_AOI_5_Khartoum_img1.tif
    # 标签: buildings_AOI_5_Khartoum_img1.geojson
    # 提取核心名称（去掉 RGB-PanSharpen_ 前缀）
    core_name = img_file.replace('RGB-PanSharpen_', '').replace('.tif', '').replace('.tiff', '')
    label_file = f"buildings_{core_name}.geojson"
    label_path = os.path.join(AOI5_LABEL_DIR, label_file)
    
    if not os.path.exists(label_path):
        # 尝试另一种可能的命名格式
        label_file_alt = f"buildings_AOI_5_Khartoum_{core_name.split('_')[-1]}.geojson"
        label_path_alt = os.path.join(AOI5_LABEL_DIR, label_file_alt)
        if os.path.exists(label_path_alt):
            label_path = label_path_alt
        else:
            print(f"    警告: 标签不存在 - {label_file}")
            continue
    
    # 提取坐标
    try:
        lon, lat = get_center_lonlat(img_path)
        country, continent = get_country_and_continent(lon, lat)
        hdi_value, hdi_level = get_hdi_info(country)
    except Exception as e:
        print(f"    坐标提取失败 ({img_file}): {e}")
        lon, lat = 32.53, 15.50
        country, continent = "Sudan", "Africa"
        hdi_value, hdi_level = 0.508, "underdeveloped"
    
    # 读取影像
    with rasterio.open(img_path) as src:
        img = src.read()
        height, width = src.height, src.width
        transform_utm = src.transform
        if img.shape[0] >= 3:
            img = img[:3, :, :]
        else:
            img = np.stack([img[0], img[0], img[0]], axis=0)
    
    # 栅格化标签
    label_mask = rasterize_geojson(label_path, transform_utm, (height, width))
    
    # 影像归一化
    img = normalize_image(img)
    
    # 切块
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            img_patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            lbl_patch = label_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            building_density = compute_building_density(lbl_patch)
            
            if np.sum(lbl_patch > 0) < MIN_BUILDING_PIXELS:
                continue
            
            settlement_type = estimate_settlement_type(building_density)
            
            img_patch_hwc = np.transpose(img_patch, (1, 2, 0))
            img_pil = Image.fromarray(img_patch_hwc)
            lbl_pil = Image.fromarray(lbl_patch)
            
            tile_name = f"SpaceNet_AOI5_{os.path.splitext(img_file)[0]}_{x}_{y}.png"
            img_pil.save(os.path.join(OUTPUT_DIR, "images", tile_name))
            lbl_pil.save(os.path.join(OUTPUT_DIR, "labels", tile_name))
            
            metadata_records.append({
                "tile_id": tile_name.replace('.png', ''),
                "image_path": f"images/{tile_name}",
                "label_path": f"labels/{tile_name}",
                "source": "SpaceNet_AOI5_Khartoum",
                "original_image": img_file,
                "resolution_m": TARGET_RESOLUTION,
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

# ----- 4. 保存元数据 -----
print("\n[4/4] 保存元数据...")
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
    print(f"覆盖国家: {df['country'].unique().tolist()}")
    print(f"大洲: {df['continent'].unique().tolist()}")
    print(f"HDI 等级: {df['hdi_level'].unique().tolist()}")
    print(f"分辨率: {df['resolution_m'].unique().tolist()} m")

print("\n处理完成！")