"""
模型定义
包含 U-Net、SegFormer、SAM 三个模型
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import numpy as np

# ============================================================
# 1. U-Net 模型
# ============================================================

class UNet(nn.Module):
    """
    U-Net with ResNet34 backbone
    二分类分割模型（背景 + 建筑）
    """
    def __init__(self, n_classes=1, backbone='resnet34', use_pretrained=False):
        super().__init__()
        encoder_weights = 'imagenet' if use_pretrained else None
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

# ============================================================
# 2. SegFormer 模型
# ============================================================

class SegFormerModel(nn.Module):
    """
    SegFormer 语义分割模型
    使用 HuggingFace 的 transformers 库
    """
    def __init__(self, n_classes=1, model_name='nvidia/mit-b2'):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            (B, 1, H, W) logits
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, n_classes, H/4, W/4)
        
        # 上采样到输入尺寸
        logits = nn.functional.interpolate(
            logits, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        return logits


# ============================================================
# 3. SAM 模型（仅推理）
# ============================================================

class SAMWrapper(nn.Module):
    """
    SAM (Segment Anything Model) 包装器
    仅用于零样本推理，不进行训练
    """
    def __init__(self, model_type='vit_b', checkpoint_path=None):
        super().__init__()
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "请安装 segment-anything: pip install segment-anything"
            )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载 SAM 模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.sam.eval()
        
        # 冻结所有参数
        for param in self.sam.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        零样本推理
        Args:
            x: (B, 3, H, W) 归一化后的图像 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        Returns:
            masks: (B, 1, H, W) 二值分割结果
        """
        from segment_anything import SamPredictor
        
        B, C, H, W = x.shape
        masks = []
        
        for i in range(B):
            # 反归一化并转换到 0-255
            img = x[i].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)
            
            predictor = SamPredictor(self.sam)
            predictor.set_image(img)
            
            # 使用图像中心点作为提示
            h, w = img.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            
            mask, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            masks.append(mask[0].astype(np.float32))
        
        masks = np.stack(masks)
        masks = torch.from_numpy(masks).unsqueeze(1).to(self.device)
        return masks


# ============================================================
# 4. 模型工厂函数
# ============================================================

def get_model(model_name, device='cuda'):
    """
    获取模型实例
    
    Args:
        model_name: 'unet', 'segformer', 或 'sam'
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: PyTorch 模型实例
    """
    if model_name == 'unet':
        model = UNet(n_classes=1, use_pretrained=False)  # 不使用预训练权重
    elif model_name == 'segformer':
        model = SegFormerModel(n_classes=1)
    elif model_name == 'sam':
        try:
            from config import SAM_CHECKPOINT, SAM_MODEL_TYPE
            model = SAMWrapper(
                model_type=SAM_MODEL_TYPE,
                checkpoint_path=SAM_CHECKPOINT
            )
        except ImportError:
            model = SAMWrapper(
                model_type='vit_b',
                checkpoint_path='sam_vit_b_01ec64.pth'
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    return model

# ============================================================
# 5. 模型参数统计
# ============================================================

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info():
    """打印各模型参数量"""
    print("=" * 50)
    print("模型参数量统计")
    print("=" * 50)
    
    # U-Net
    unet = UNet(n_classes=1)
    print(f"U-Net (ResNet34):     {count_parameters(unet):,} 参数")
    
    # SegFormer
    try:
        segformer = SegFormerModel(n_classes=1)
        print(f"SegFormer (MIT-B2):   {count_parameters(segformer):,} 参数")
    except Exception as e:
        print(f"SegFormer: 加载失败 ({e})")
    
    # SAM (不计算参数，因为只推理)
    print(f"SAM (ViT-B):          ~91M 参数 (仅推理)")
    
    print("=" * 50)


if __name__ == "__main__":
    get_model_info()