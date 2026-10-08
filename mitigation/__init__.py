"""
地理公平性优化策略模块
"""

from .gw_loss import CombinedLoss, GeographicallyWeightedLoss
from .gss import GeographicallyStratifiedSampler, create_balanced_dataloader, get_region_weights_from_performance
from .ada import GradientReversalLayer, DomainClassifier, AdversarialDomainAdaptationModel, domain_adaptation_loss
from .gat import GeographicallyAdaptiveThresholding, adaptive_threshold_postprocessing

__all__ = [
    'CombinedLoss',
    'GeographicallyWeightedLoss',
    'GeographicallyStratifiedSampler',
    'create_balanced_dataloader',
    'get_region_weights_from_performance',
    'GradientReversalLayer',
    'DomainClassifier',
    'AdversarialDomainAdaptationModel',
    'domain_adaptation_loss',
    'GeographicallyAdaptiveThresholding',
    'adaptive_threshold_postprocessing'
]