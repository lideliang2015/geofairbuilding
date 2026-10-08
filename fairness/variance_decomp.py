"""
多层次方差分解模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def multilevel_variance_decomposition(
    df: pd.DataFrame,
    iou_col: str = 'iou',
    level_cols: List[str] = None
) -> Dict[str, float]:
    """
    多层次方差分解 (ANOVA方法)
    """
    if level_cols is None:
        level_cols = ['continent', 'country', 'city', 'tile']
    
    iou_values = df[iou_col].values
    total_var = np.var(iou_values, ddof=1)
    
    proportions = {}
    remaining_var = total_var
    
    for i, level in enumerate(level_cols):
        if level not in df.columns:
            proportions[level] = 0.0
            continue
            
        if i == len(level_cols) - 1:
            proportions[level] = remaining_var / total_var if total_var > 0 else 0.0
        else:
            group_means = df.groupby(level)[iou_col].mean()
            group_var = np.var(group_means.values, ddof=1) if len(group_means) > 1 else 0
            between_var = group_var * (len(group_means) - 1) / len(df) if len(df) > 0 else 0
            proportions[level] = between_var / total_var if total_var > 0 else 0.0
            remaining_var -= between_var
    
    return proportions


def compute_variance_confidence_intervals(
    df: pd.DataFrame,
    iou_col: str = 'iou',
    level_cols: List[str] = None,
    n_bootstrap: int = 100,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap置信区间"""
    np.random.seed(random_state)
    proportions_list = []
    
    for _ in range(n_bootstrap):
        boot_df = df.sample(n=len(df), replace=True)
        props = multilevel_variance_decomposition(boot_df, iou_col, level_cols)
        proportions_list.append(props)
    
    ci = {}
    if level_cols is None:
        level_cols = ['continent', 'country', 'city', 'tile']
    
    for level in level_cols:
        values = [p.get(level, 0) for p in proportions_list]
        ci[level] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
    
    return ci


def prepare_dataframe_for_decomposition(
    iou_values: List[float],
    continent_ids: List[int],
    country_ids: List[int],
    city_ids: List[int],
    tile_ids: List[int]
) -> pd.DataFrame:
    """准备DataFrame"""
    return pd.DataFrame({
        'iou': iou_values,
        'continent': continent_ids,
        'country': country_ids,
        'city': city_ids,
        'tile': tile_ids
    })