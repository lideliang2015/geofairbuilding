"""
空间统计模块 - 纯NumPy实现
提供 Moran's I, Getis-Ord Gi*, 空间权重矩阵等空间统计功能
"""

import numpy as np
from typing import Tuple


def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    计算欧氏距离矩阵
    
    Args:
        coordinates: 坐标数组 (N, 2)，格式为 [[lon1, lat1], [lon2, lat2], ...]
    
    Returns:
        dist_matrix: 距离矩阵 (N, N)
    """
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def compute_spatial_weight_matrix(
    coordinates: np.ndarray,
    threshold_km: float = 50.0,
    binary: bool = True
) -> np.ndarray:
    """
    计算空间权重矩阵（考虑地理距离）
    
    Args:
        coordinates: 坐标数组 (N, 2)，格式为 [[lon, lat], ...]
        threshold_km: 空间权重阈值（公里）
        binary: True=二进制权重（距离阈值内为1），False=距离倒数权重
    
    Returns:
        w: 行标准化的空间权重矩阵 (N, N)
    """
    # 坐标缩放：1度纬度 ≈ 111公里，1度经度 ≈ 111 * cos(纬度) 公里
    lat_scale = 111.0
    avg_lat = np.mean(coordinates[:, 1])
    lon_scale = 111.0 * np.cos(np.radians(avg_lat))
    
    scaled_coords = coordinates.copy()
    scaled_coords[:, 0] *= lon_scale
    scaled_coords[:, 1] *= lat_scale
    
    n = len(coordinates)
    w = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((scaled_coords[i] - scaled_coords[j]) ** 2))
                if dist < threshold_km:
                    if binary:
                        w[i, j] = 1.0
                    else:
                        w[i, j] = 1.0 / (dist + 1e-6)
    
    # 行标准化
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    w = w / row_sums
    
    return w


def compute_global_moran_i(
    values: np.ndarray,
    coordinates: np.ndarray,
    threshold_km: float = 50.0
) -> Tuple[float, float]:
    """
    计算全局 Moran's I 指数
    
    Args:
        values: 数值数组 (N,)
        coordinates: 坐标数组 (N, 2)
        threshold_km: 空间权重阈值（公里）
    
    Returns:
        (Moran's I, 期望值)
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    
    # 计算空间权重矩阵
    w = compute_spatial_weight_matrix(coordinates, threshold_km, binary=True)
    
    # 计算偏差
    z = values - np.mean(values)
    
    # 计算Moran's I
    numerator = np.sum(w * np.outer(z, z))
    denominator = np.sum(z ** 2)
    
    if denominator == 0:
        return 0.0, -1.0 / (n - 1)
    
    moran_i = (n / np.sum(w)) * (numerator / denominator)
    expected = -1.0 / (n - 1)
    
    return moran_i, expected


def compute_getis_ord_gi_star(
    values: np.ndarray,
    coordinates: np.ndarray,
    threshold_km: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Getis-Ord Gi* 统计量（包含自身）
    
    Args:
        values: 数值数组 (N,)
        coordinates: 坐标数组 (N, 2)
        threshold_km: 空间权重阈值（公里）
    
    Returns:
        (Gi*统计量数组, Z-score数组)
    """
    n = len(values)
    w = compute_spatial_weight_matrix(coordinates, threshold_km, binary=True)
    
    # 全局均值和标准差
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    gi_star = np.zeros(n)
    z_scores = np.zeros(n)
    
    for i in range(n):
        # Gi* 包含自身，所以将自身权重设为1
        w_i = w[i].copy()
        w_i[i] = 1.0
        
        # 加权和
        sum_wx = np.sum(w_i * values)
        sum_w = np.sum(w_i)
        
        # 期望和方差
        expected = sum_w * mean_val
        variance = sum_w ** 2 * std_val ** 2
        
        if variance > 0:
            gi_star[i] = (sum_wx - expected) / np.sqrt(variance)
            z_scores[i] = gi_star[i]
        else:
            gi_star[i] = 0.0
            z_scores[i] = 0.0
    
    return gi_star, z_scores


def identify_hotspots(
    z_scores: np.ndarray,
    p_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    识别热点和冷点区域
    
    Args:
        z_scores: Z-score数组 (N,)
        p_threshold: 显著性阈值（默认0.05，对应z≈1.96）
    
    Returns:
        (low_hotspots, high_hotspots): 低值热点和高值热点的布尔数组
    """
    # 对于p=0.05，双侧检验对应z阈值约为1.96
    z_threshold = 1.96
    
    # 低值热点（不公平热点）：z < -1.96
    low_hotspots = z_scores < -z_threshold
    
    # 高值热点（公平性好的区域）：z > 1.96
    high_hotspots = z_scores > z_threshold
    
    return low_hotspots, high_hotspots


def compute_performance_heatmap(
    iou_values: np.ndarray,
    coordinates: np.ndarray,
    grid_size: Tuple[int, int] = (100, 100),
    method: str = 'linear'
) -> np.ndarray:
    """
    生成性能热力图（空间插值）
    
    Args:
        iou_values: IoU值数组 (N,)
        coordinates: 坐标数组 (N, 2)
        grid_size: 网格大小 (height, width)
        method: 插值方法 ('linear', 'cubic', 'nearest')
    
    Returns:
        heatmap: 热力图矩阵 (grid_size[0], grid_size[1])
    """
    from scipy.interpolate import griddata
    
    # 确定网格范围
    lon_min, lon_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    lat_min, lat_max = coordinates[:, 1].min(), coordinates[:, 1].max()
    
    # 添加边距
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05
    
    lon_min -= lon_padding
    lon_max += lon_padding
    lat_min -= lat_padding
    lat_max += lat_padding
    
    # 创建网格
    lon_grid = np.linspace(lon_min, lon_max, grid_size[1])
    lat_grid = np.linspace(lat_min, lat_max, grid_size[0])
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # 插值
    heatmap = griddata(
        coordinates, iou_values, (lon_mesh, lat_mesh),
        method=method, fill_value=np.nan
    )
    
    return heatmap


def compute_spatial_lag(
    values: np.ndarray,
    coordinates: np.ndarray,
    threshold_km: float = 50.0
) -> np.ndarray:
    """
    计算空间滞后（邻居的加权平均值）
    
    Args:
        values: 数值数组 (N,)
        coordinates: 坐标数组 (N, 2)
        threshold_km: 空间权重阈值（公里）
    
    Returns:
        spatial_lag: 空间滞后值数组 (N,)
    """
    w = compute_spatial_weight_matrix(coordinates, threshold_km, binary=False)
    spatial_lag = w @ values
    return spatial_lag