import os
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio import features
from PIL import Image
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
IMG_DIR = r"E:\03.spacenet\AOI_1_Rio\PS-RGB"
LABEL_FILE = r"E:\03.spacenet\AOI_1_Rio\geojson_buildings\AOI_1_Rio_geojson_buildings.geojson"
OUTPUT_DIR = r"E:\processed\05.spacenet_rio"

PATCH_SIZE = 512
STRIDE = 256
MIN_BUILDING_PIXELS = 100

# 巴西 HDI 信息
COUNTRY = "Brazil"
CONTINENT = "South America"
HDI_VALUE = 0.754
HDI_LEVEL = "developing"  # 巴西 HDI 0.754，属于发展中
RESOLUTION = 0.5
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print("=" * 60)
print("SpaceNet AOI_1_Rio 预处理脚本")
print("=" * 60)

# ----- 1. 加载建筑标签 -----
print("\n[1/3] 加载建筑标签...")
gdf_all = gpd.read_file(LABEL_FILE)
print(f"    总建筑物数: {len(gdf_all)}")
print(f"    原始 CRS: {gdf_all.crs}")

# ----- 2. 处理每个影像 -----
print("\n[2/3] 处理影像...")

image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.tif')]
print(f"    找到 {len(image_files)} 个影像文件")

metadata_records = []

for img_file in tqdm(image_files, desc="处理进度"):
    img_path = os.path.join(IMG_DIR, img_file)
    
    # 读取影像信息
    with rasterio.open(img_path) as src:
        img = src.read()
        height, width = src.height, src.width
        transform_utm = src.transform
        crs = src.crs
        bounds = src.bounds
    
    # 获取影像中心点坐标（用于元数据）
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.bottom + bounds.top) / 2
    lon, lat = transform(crs, 'EPSG:4326', [center_x], [center_y])
    lon, lat = lon[0], lat[0]
    
    # 筛选该影像范围内的建筑
    img_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # 先转换 gdf_all 到影像的坐标系（只转换一次）
    if not hasattr(gdf_all, '_cached_crs') or gdf_all._cached_crs != crs:
        gdf_crs = gdf_all.to_crs(crs)
        gdf_all._cached_crs = crs
        gdf_all._cached_gdf = gdf_crs
    
    gdf_clipped = gdf_all._cached_gdf[gdf_all._cached_gdf.geometry.intersects(img_bbox)]
    
    if len(gdf_clipped) == 0:
        continue
    
    # 栅格化
    shapes = ((geom, 1) for geom in gdf_clipped.geometry)
    label_mask = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform_utm,
        fill=0,
        dtype=np.uint8
    )
    label_mask = (label_mask > 0).astype(np.uint8) * 255
    
    # 影像归一化
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
    
    # 保留 RGB 波段
    if img.shape[0] > 3:
        img = img[:3, :, :]
    elif img.shape[0] == 1:
        img = np.stack([img[0], img[0], img[0]], axis=0)
    
    # 切块
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            img_patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            lbl_patch = label_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            building_pixels = np.sum(lbl_patch > 0)
            if building_pixels < MIN_BUILDING_PIXELS:
                continue
            
            img_patch_hwc = np.transpose(img_patch, (1, 2, 0))
            img_pil = Image.fromarray(img_patch_hwc)
            lbl_pil = Image.fromarray(lbl_patch)
            
            tile_name = f"SpaceNet_RIO_{os.path.splitext(img_file)[0]}_{x}_{y}.png"
            img_pil.save(os.path.join(OUTPUT_DIR, "images", tile_name))
            lbl_pil.save(os.path.join(OUTPUT_DIR, "labels", tile_name))
            
            metadata_records.append({
                "tile_id": tile_name.replace('.png', ''),
                "image_path": f"images/{tile_name}",
                "label_path": f"labels/{tile_name}",
                "source": "SpaceNet_RIO",
                "original_image": img_file,
                "resolution_m": RESOLUTION,
                "width": PATCH_SIZE,
                "height": PATCH_SIZE,
                "x_offset": x,
                "y_offset": y,
                "longitude": lon,
                "latitude": lat,
                "country": COUNTRY,
                "continent": CONTINENT,
                "hdi_value": HDI_VALUE,
                "hdi_level": HDI_LEVEL,
                "building_density": building_pixels / (PATCH_SIZE * PATCH_SIZE),
                "settlement_type": "urban"
            })

# ----- 3. 保存元数据 -----
print("\n[3/3] 保存元数据...")
df = pd.DataFrame(metadata_records)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print(f"\n✅ 预处理完成！")
print(f"    - 输出目录: {OUTPUT_DIR}")
print(f"    - 生成瓦片数: {len(metadata_records)}")
print(f"    - 元数据文件: {OUTPUT_DIR}/metadata.csv")

# 统计信息
if len(df) > 0:
    print("\n=== 统计信息 ===")
    print(f"国家: {df['country'].unique().tolist()}")
    print(f"大洲: {df['continent'].unique().tolist()}")
    print(f"HDI 等级: {df['hdi_level'].unique().tolist()}")
    print(f"分辨率: {df['resolution_m'].unique().tolist()} m")