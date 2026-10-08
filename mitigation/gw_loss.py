"""
地理加权损失 (Geographically Weighted Loss)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class CombinedLoss(nn.Module):
    """交叉熵 + Dice 联合损失"""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W]
            targets: [B, H, W]
            reduction: 'mean' or 'none'
        """
        # 交叉熵损失
        ce = self.ce_loss(predictions, targets)
        
        # Dice损失
        preds_softmax = torch.softmax(predictions, dim=1)
        preds_binary = preds_softmax[:, 1, :, :]
        
        # 平滑处理
        smooth = 1e-6
        
        intersection = (preds_binary * targets.float()).sum(dim=(1, 2))
        union = preds_binary.sum(dim=(1, 2)) + targets.float().sum(dim=(1, 2))
        
        dice_per_sample = 1 - (2 * intersection + smooth) / (union + smooth)
        
        if reduction == 'mean':
            ce = ce.mean()
            dice = dice_per_sample.mean()
        else:
            dice = dice_per_sample
        
        return self.ce_weight * ce + self.dice_weight * dice


class GeographicallyWeightedLoss(nn.Module):
    """
    地理加权损失
    
    权重计算: w_i = (1 / P_i)^temperature
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        performance_map: Dict[str, float],
        region_to_id: Dict[str, int],
        temperature: float = 1.0,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.base_loss = base_loss
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 计算权重
        n_regions = len(region_to_id)
        raw_weights = np.zeros(n_regions)
        
        for region, region_id in region_to_id.items():
            perf = performance_map.get(region, 0.5)
            raw_weights[region_id] = 1.0 / (perf + 1e-6)
        
        # 归一化
        normalized_weights = raw_weights / raw_weights.mean()
        scaled_weights = normalized_weights ** temperature
        clipped_weights = np.clip(scaled_weights, min_weight, max_weight)
        
        # 注册为buffer
        self.register_buffer('weight_tensor', torch.tensor(clipped_weights, dtype=torch.float32, device=device))
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        region_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: 模型输出 [B, C, H, W]
            targets: 真值标签 [B, H, W]
            region_indices: 每个样本的区域索引 [B]
        
        Returns:
            weighted_loss: 加权后的损失
        """
        # 计算每个像素的基础损失
        base_losses = self.base_loss(predictions, targets, reduction='none')
        
        # 获取每个样本的权重
        sample_weights = self.weight_tensor[region_indices]
        
        # 扩展为像素级权重
        pixel_weights = sample_weights.view(-1, 1, 1)
        
        # 加权平均
        weighted_loss = (base_losses * pixel_weights).mean()
        
        return weighted_loss