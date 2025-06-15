#!/usr/bin/env python3
"""
扩散模型训练启动脚本
集成数据加载、模型构建和训练流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path

# 导入项目模块
from config.model_config import DiffusionConfig, TrainingConfig
from models.diffusion.ddpm import DiffusionModel
from models.diffusion.unet import UNet
from data.datasets.image_dataset import ImageFolderDataset, TextConditionedDataset
from training.trainer import Trainer


def create_transforms(image_size=256):
    """创建图像变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])


def setup_cifar10_dataset(data_dir, batch_size=32, image_size=32):
    """设置CIFAR-10数据集"""
    import torchvision
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def setup_custom_dataset(data_dir, batch_size=32, image_size=256):
    """设置自定义数据集"""
    transform = create_transforms(image_size)
    
    # 训练集
    train_dir = Path(data_dir) / "custom" / "train"
    if not train_dir.exists():
        raise ValueError(f"训练数据目录不存在: {train_dir}")
    
    train_dataset = ImageFolderDataset(
        root_dir=str(train_dir),
        transform=transform,
        image_size=image_size
    )
    
    # 验证集
    val_dir = Path(data_dir) / "custom" / "val"
    val_dataset = None
    if val_dir.exists():
        val_dataset = ImageFolderDataset(
            root_dir=str(val_dir),
            transform=transform,
            image_size=image_size
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def create_model(model_config, image_channels=3):
    """创建扩散模型"""
    # 创建UNet噪声预测器
    unet = UNet(
        dim=model_config.model_dim,
        channels=image_channels,
        dim_mults=model_config.unet_dim_mults,
        resnet_block_groups=model_config.unet_resnet_blocks,
        use_convnext=True,
        convnext_mult=2,
    )
    
    # 创建扩散模型
    diffusion_model = DiffusionModel(
        noise_predictor=unet,
        config=model_config
    )
    
    return diffusion_model


def main():
    parser = argparse.ArgumentParser(description="训练扩散模型")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "custom"], 
                       default="cifar10", help="选择数据集")
    parser.add_argument("--data_dir", type=str, default="./data/raw", 
                       help="数据目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--image_size", type=int, default=32,
                       help="图像大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--device", type=str, default="auto",
                       help="训练设备")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建配置
    model_config = DiffusionConfig()
    train_config = TrainingConfig()
    
    # 更新配置
    train_config.batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    
    # 根据数据集调整图像大小
    if args.dataset == "cifar10":
        args.image_size = 32
        image_channels = 3
    else:
        image_channels = 3
    
    print(f"图像大小: {args.image_size}x{args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    
    # 设置数据集
    print(f"准备数据集: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, val_loader = setup_cifar10_dataset(
            args.data_dir, args.batch_size, args.image_size
        )
    elif args.dataset == "custom":
        train_loader, val_loader = setup_custom_dataset(
            args.data_dir, args.batch_size, args.image_size
        )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    if val_loader:
        print(f"验证样本数: {len(val_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = create_model(model_config, image_channels)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config.__dict__,
        device=device,
        save_dir=args.save_dir
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("开始训练...")
    try:
        trainer.train(num_epochs=args.epochs)
    except KeyboardInterrupt:
        print("训练被用户中断")
        # 保存当前状态
        checkpoint_path = Path(args.save_dir) / "checkpoint_interrupted.pt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"已保存中断时的检查点: {checkpoint_path}")
    
    print("训练完成！")


if __name__ == "__main__":
    main() 