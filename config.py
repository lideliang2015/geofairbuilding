"""
配置文件（优化版）
"""

import os

# ========== 路径配置 ==========
DATASET_ROOT = r"E:\GeoFair-Building-v1.0"
OUTPUT_DIR = r"E:\experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 训练参数（优化）==========
BATCH_SIZE = 16
NUM_EPOCHS = 100           # 增加到 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4        # 添加权重衰减
NUM_WORKERS = 0
DEVICE = 'cuda'

# 损失函数权重
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# 学习率调度
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.5

# 早停
EARLY_STOP_PATIENCE = 20   # 增加到 20

# 随机种子
RANDOM_SEED = 42

# ========== 模型配置 ==========
UNET_BACKBONE = 'resnet34'
SEGFORMER_VERSION = 'nvidia/mit-b2'
SAM_MODEL_TYPE = 'vit_b'
SAM_CHECKPOINT = r"E:\geofair_experiments\sam_vit_b_01ec64.pth"

# ========== 评估参数 ==========
EVALUATION_LEVELS = ['continent', 'hdi_level']