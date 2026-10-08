"""
公平性评估模块
"""

from .metrics import (
    compute_iou_per_sample,
    compute_regional_performance,
    compute_spatial_fairness_ratio,
    compute_spatial_fairness_index,
    compute_equalized_odds_difference,
    compute_hotspot_ratio
)

from .variance_decomp import (
    multilevel_variance_decomposition,
    compute_variance_confidence_intervals,
    prepare_dataframe_for_decomposition
)

from .spatial_stats import (
    compute_global_moran_i,
    compute_getis_ord_gi_star,
    identify_hotspots,
    compute_performance_heatmap,
    compute_spatial_weight_matrix
)

__all__ = [
    'compute_iou_per_sample',
    'compute_regional_performance',
    'compute_spatial_fairness_ratio',
    'compute_spatial_fairness_index',
    'compute_equalized_odds_difference',
    'compute_hotspot_ratio',
    'multilevel_variance_decomposition',
    'compute_variance_confidence_intervals',
    'prepare_dataframe_for_decomposition',
    'compute_global_moran_i',
    'compute_getis_ord_gi_star',
    'identify_hotspots',
    'compute_performance_heatmap',
    'compute_spatial_weight_matrix'
]