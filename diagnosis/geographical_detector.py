"""
地理探测器 (Geographical Detector)
用于识别地理不公平的驱动因素
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def compute_q_statistic(
    y: np.ndarray,
    x: np.ndarray,
    n_strata: int = 5
) -> Tuple[float, float]:
    """
    计算地理探测器的 q 统计量
    
    Args:
        y: 性能值数组 (N,)
        x: 驱动因素值数组 (N,)
        n_strata: 分层数量
    
    Returns:
        (q_statistic, p_value)
    """
    # 将驱动因素离散化为分层
    x_discrete = pd.cut(x, bins=n_strata, labels=False)
    
    # 全局方差
    N = len(y)
    var_global = np.var(y, ddof=1)
    
    # 层内方差加权和
    var_within_sum = 0
    unique_strata = np.unique(x_discrete)
    
    for s in unique_strata:
        y_s = y[x_discrete == s]
        if len(y_s) > 1:
            var_within_sum += len(y_s) * np.var(y_s, ddof=1)
    
    # q统计量
    q = 1 - var_within_sum / (N * var_global)
    
    # 简化的p值估计（非中心F分布近似）
    # 自由度
    L = len(unique_strata)
    df1 = L - 1
    df2 = N - L
    
    # 非中心参数
    lambda_nc = q * df1 / (1 - q) if q < 1 else 1e10
    
    # 使用F分布近似
    from scipy.stats import f
    f_stat = (q / (1 - q)) * (df2 / df1) if q < 1 else np.inf
    p_value = 1 - f.cdf(f_stat, df1, df2) if f_stat != np.inf else 0.0
    
    return q, p_value


def geographical_detector(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    n_strata: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    对多个驱动因素进行地理探测器分析
    
    Args:
        df: 包含性能值和驱动因素的DataFrame
        y_col: 性能值列名
        x_cols: 驱动因素列名列表
        n_strata: 分层数量
    
    Returns:
        各驱动因素的q值和p值
    """
    results = {}
    y = df[y_col].values
    
    for x_col in x_cols:
        x = df[x_col].values
        q, p = compute_q_statistic(y, x, n_strata)
        results[x_col] = {'q': q, 'p_value': p}
    
    # 按q值排序
    results = dict(sorted(results.items(), key=lambda item: item[1]['q'], reverse=True))
    
    return results


def geodetector_attribution(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    n_strata: int = 5
) -> Dict[str, float]:
    """
    归因分析：计算各驱动因素对性能差距的解释比例
    
    Args:
        df: 数据框
        y_col: 性能值列名
        x_cols: 驱动因素列名列表
        n_strata: 分层数量
    
    Returns:
        各驱动因素的解释比例
    """
    # 计算各因素的q值
    q_results = geographical_detector(df, y_col, x_cols, n_strata)
    
    # 归一化q值作为解释比例
    total_q = sum([r['q'] for r in q_results.values()])
    
    if total_q == 0:
        return {k: 0 for k in q_results.keys()}
    
    attribution = {k: r['q'] / total_q for k, r in q_results.items()}
    
    return attribution