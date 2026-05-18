"""
数据集加载器（修复元数据传递）
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class GeoFairDataset(Dataset):
    """GeoFair-Building v1.0 数据集"""
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        
        # 读取元数据
        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        
        # 读取划分文件
        splits_path = os.path.join(root_dir, "splits.csv")
        if os.path.exists(splits_path):
            splits = pd.read_csv(splits_path)
            split_ids = splits[splits['split'] == split]['tile_id'].tolist()
            self.metadata = self.metadata[self.metadata['tile_id'].isin(split_ids)].reset_index(drop=True)
        
        self.transform = transform
        self.target_transform = target_transform
        
        print(f"加载 {split} 集: {len(self.metadata)} 个样本")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 读取影像
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # 读取标签
        lbl_path = os.path.join(self.root_dir, row['label_path'])
        label = Image.open(lbl_path).convert('L')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = transforms.ToTensor()(label)
            label = (label > 0.5).float()
        
        # 元数据（确保正确传递）
        metadata = {
            'tile_id': row['tile_id'],
            'continent': row['continent'] if pd.notna(row['continent']) else 'Unknown',
            'hdi_level': row['hdi_level'] if pd.notna(row['hdi_level']) else 'Unknown',
            'resolution_m': float(row['resolution_m']) if pd.notna(row['resolution_m']) else 0.5,
            'source': row['source'] if pd.notna(row['source']) else 'Unknown'
        }
        
        return image, label, metadata


def get_transforms():
    """获取数据增强变换"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_dataloaders(root_dir, batch_size, num_workers=4):
    """获取训练、验证、测试数据加载器"""
    train_transform, val_transform = get_transforms()
    
    train_dataset = GeoFairDataset(root_dir, split='train', transform=train_transform)
    val_dataset = GeoFairDataset(root_dir, split='val', transform=val_transform)
    test_dataset = GeoFairDataset(root_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_test_loader(root_dir, batch_size, num_workers=4):
    """仅获取测试集加载器"""
    _, val_transform = get_transforms()
    test_dataset = GeoFairDataset(root_dir, split='test', transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return test_loader, test_dataset.metadata