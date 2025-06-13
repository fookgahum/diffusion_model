#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
扩散模型训练脚本

提供了完整的扩散模型训练流程，包括:
- 数据加载和预处理
- 模型构建
- 训练循环
- 模型保存
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from models.diffusion.unet import UNet
from models.diffusion.ddpm import DiffusionModel
from data.datasets.image_dataset import ImageFolderDataset, get_dataloaders
from training.trainer import Trainer
from config.model_config import DiffusionConfig, TrainingConfig


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练扩散模型")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='训练数据目录')
    parser.add_argument('--condition_dir', type=str, default=None,
                        help='条件数据目录（如果是条件模型）')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    # 模型参数
    parser.add_argument('--model_dim', type=int, default=128,
                        help='模型基础通道数')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='扩散步数')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='beta调度类型')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练周期数')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每多少个周期保存一次模型')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作进程数')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备(cuda或cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def main():
    """主训练流程"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 配置模型和训练参数
    logger.info("配置模型参数...")
    model_config = DiffusionConfig()
    model_config.model_dim = args.model_dim
    model_config.timesteps = args.timesteps
    model_config.beta_schedule = args.beta_schedule
    
    train_config = TrainingConfig()
    train_config.batch_size = args.batch_size
    train_config.learning_rate = args.lr
    train_config.weight_decay = args.weight_decay
    
    # 创建数据集和数据加载器
    logger.info("加载数据集...")
    dataset = ImageFolderDataset(
        root_dir=args.data_dir,
        condition_dir=args.condition_dir,
        image_size=args.image_size
    )
    
    train_loader, val_loader = get_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    logger.info(f"训练集大小: {len(train_loader.sampler)}")
    logger.info(f"验证集大小: {len(val_loader.sampler)}")
    
    # 创建模型
    logger.info("构建模型...")
    unet = UNet(
        in_channels=3,
        model_channels=model_config.model_dim,
        out_channels=3,
        time_emb_dim=model_config.time_emb_dim
    )
    
    model = DiffusionModel(unet, model_config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        save_dir=args.save_dir
    )
    
    # 恢复训练(如果指定了检查点)
    if args.resume:
        logger.info(f"恢复训练自: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(
        trainer.optimizer, 
        T_max=args.epochs * len(train_loader),
        eta_min=1e-6
    )
    trainer.scheduler = scheduler
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(num_epochs=args.epochs)
    
    logger.info("训练完成!")


if __name__ == "__main__":
    main() 