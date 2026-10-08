"""
地理偏差诊断模块 (RQ3)
"""

from .geographical_detector import geographical_detector, compute_q_statistic
from .mgwr import MultiScaleGeographicallyWeightedRegression

__all__ = [
    'geographical_detector',
    'compute_q_statistic',
    'MultiScaleGeographicallyWeightedRegression'
]