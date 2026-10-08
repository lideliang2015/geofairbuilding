"""
多尺度地理加权回归 (Multi-Scale Geographically Weighted Regression)
简化实现
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class MultiScaleGeographicallyWeightedRegression:
    """
    多尺度地理加权回归的简化实现
    
    用于分析驱动因素影响的空间异质性
    """
    
    def __init__(self, kernel: str = 'gaussian', bandwidth: Optional[float] = None):
        """
        Args:
            kernel: 核函数类型 ('gaussian', 'bisquare')
            bandwidth: 带宽（None=自适应）
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.coefficients_ = None
        self.global_coefficients_ = None
    
    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def _compute_weights(self, dist_matrix: np.ndarray, bandwidth: float) -> np.ndarray:
        """计算权重矩阵"""
        n = len(dist_matrix)
        w = np.zeros((n, n))
        
        if self.kernel == 'gaussian':
            # 高斯核
            for i in range(n):
                w[i] = np.exp(-0.5 * (dist_matrix[i] / bandwidth) ** 2)
        elif self.kernel == 'bisquare':
            # 双平方核
            for i in range(n):
                u = dist_matrix[i] / bandwidth
                w[i] = (1 - u ** 2) ** 2 * (u < 1)
        else:
            # 默认：均匀核
            for i in range(n):
                w[i] = (dist_matrix[i] < bandwidth).astype(float)
        
        return w
    
    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray) -> 'MultiScaleGeographicallyWeightedRegression':
        """
        拟合MGWR模型
        
        Args:
            X: 特征矩阵 (N, p)
            y: 目标变量 (N,)
            coords: 坐标 (N, 2)
        
        Returns:
            self
        """
        n, p = X.shape
        dist_matrix = self._compute_distance_matrix(coords)
        
        # 自适应带宽（中位数距离）
        if self.bandwidth is None:
            self.bandwidth = np.median(dist_matrix[dist_matrix > 0]) * 2
        
        # 计算每个位置的局部系数
        coefficients = np.zeros((n, p))
        
        for i in range(n):
            # 计算权重
            weights = self._compute_weights(dist_matrix[i], self.bandwidth)
            W = np.diag(weights)
            
            # 加权最小二乘
            XtW = X.T @ W
            try:
                coeff = np.linalg.solve(XtW @ X, XtW @ y)
                coefficients[i] = coeff
            except np.linalg.LinAlgError:
                # 使用伪逆
                coeff = np.linalg.pinv(XtW @ X) @ XtW @ y
                coefficients[i] = coeff
        
        self.coefficients_ = coefficients
        
        # 全局系数（普通最小二乘）
        self.global_coefficients_ = np.linalg.lstsq(X, y, rcond=None)[0]
        
        return self
    
    def get_coefficient_variation(self) -> np.ndarray:
        """
        计算系数的空间变异系数
        
        Returns:
            coefficient_variation: 每个特征的变异系数
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet")
        
        stds = np.std(self.coefficients_, axis=0)
        means = np.abs(np.mean(self.coefficients_, axis=0))
        
        cv = stds / (means + 1e-8)
        
        return cv
    
    def get_spatial_patterns(self) -> Dict[str, np.ndarray]:
        """
        获取空间模式描述
        
        Returns:
            包含各特征系数分布的字典
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet")
        
        patterns = {}
        for i in range(self.coefficients_.shape[1]):
            coeff = self.coefficients_[:, i]
            patterns[f'feature_{i}'] = {
                'mean': np.mean(coeff),
                'std': np.std(coeff),
                'min': np.min(coeff),
                'max': np.max(coeff),
                'cv': np.std(coeff) / (np.abs(np.mean(coeff)) + 1e-8)
            }
        
        return patterns