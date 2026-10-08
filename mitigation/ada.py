"""
对抗域适应 (Adversarial Domain Adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层"""
    
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """梯度反转层的封装"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """地理区域分类器"""
    
    def __init__(self, feature_dim: int, n_domains: int, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_domains)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AdversarialDomainAdaptationModel(nn.Module):
    """对抗域适应模型包装器"""
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        segmenter: nn.Module,
        n_domains: int,
        alpha: float = 0.1
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.segmenter = segmenter
        self.grl = GradientReversalLayer(alpha=alpha)
        
        # 获取特征维度
        feature_dim = self._get_feature_dim()
        self.domain_classifier = DomainClassifier(feature_dim, n_domains)
    
    def _get_feature_dim(self) -> int:
        """获取特征提取器输出的特征维度"""
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            try:
                features = self.feature_extractor(dummy_input)
                # 处理不同类型的输出
                if isinstance(features, tuple):
                    features = features[0]
                return features.shape[1]
            except Exception:
                return 256  # 默认值
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_features: 是否返回中间特征
        
        Returns:
            seg_output: 分割输出 [B, 2, H, W]
            domain_logits: 区域分类输出 [B, n_domains]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 处理不同类型的输出
        if isinstance(features, tuple):
            features = features[0]
        
        # 全局平均池化用于区域分类
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        # 梯度反转后输入区域分类器
        reversed_features = self.grl(pooled)
        domain_logits = self.domain_classifier(reversed_features)
        
        # 分割输出
        seg_output = self.segmenter(features)
        
        if return_features:
            return seg_output, domain_logits, features
        
        return seg_output, domain_logits


def domain_adaptation_loss(
    seg_output: torch.Tensor,
    seg_target: torch.Tensor,
    domain_logits: torch.Tensor,
    domain_target: torch.Tensor,
    lambda_adv: float = 0.1,
    ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对抗域适应总损失
    
    Args:
        seg_output: 分割预测 [B, C, H, W]
        seg_target: 分割真值 [B, H, W]
        domain_logits: 区域分类预测 [B, n_domains]
        domain_target: 区域真值 [B]
        lambda_adv: 对抗损失权重
        ignore_index: 忽略的索引
    
    Returns:
        total_loss, seg_loss, domain_loss
    """
    # 分割损失
    seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(seg_output, seg_target)
    
    # 区域分类损失
    domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_target)
    
    # 总损失
    total_loss = seg_loss + lambda_adv * domain_loss
    
    return total_loss, seg_loss, domain_loss