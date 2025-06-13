"""
扩散模型训练器

实现了扩散模型的训练循环，包括:
- 数据加载和预处理
- 优化器和学习率调度
- 模型参数更新
- 训练状态跟踪和检查点保存
- EMA(指数移动平均)模型维护
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Optional, Any, Tuple

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EMA:
    """
    指数移动平均模型
    用于保持一个更稳定的模型版本
    """
    def __init__(self, model, decay=0.9999):
        """
        初始化EMA
        
        参数:
            model: 要跟踪的模型
            decay: EMA衰减率
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 注册参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    扩散模型训练器
    """
    def __init__(self, 
                 model,                 # 模型实例
                 train_loader,          # 训练数据加载器
                 val_loader=None,       # 验证数据加载器
                 scheduler=None,        # 学习率调度器
                 config=None,           # 训练配置
                 device=None,           # 训练设备
                 save_dir="checkpoints" # 保存目录
                ):
        """
        初始化训练器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 确保配置存在
        self.config = config or {}
        
        # 设置设备
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # EMA模型
        self.use_ema = self.config.get("use_ema", True)
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.config.get("ema_decay", 0.9999))
        
        # 检查点保存
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 训练状态
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        self.losses = []
        
    def _create_optimizer(self):
        """创建优化器"""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        return optimizer
    
    def save_checkpoint(self, path, is_best=False):
        """保存检查点"""
        state = {
            'epoch': self.epoch,
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 保存EMA模型
        if self.use_ema:
            state['ema'] = self.ema.shadow
            
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")
        
        # 如果是最佳模型，复制一份
        if is_best:
            best_path = os.path.join(os.path.dirname(path), "model_best.pt")
            torch.save(state, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.losses = checkpoint.get('losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # 加载EMA
        if self.use_ema and 'ema' in checkpoint:
            self.ema.shadow = checkpoint['ema']
            
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        
    def train_epoch(self, epoch):
        """训练一个周期"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 将数据移动到设备
            if isinstance(batch, dict):
                # 字典类型批次 (如带条件信息)
                x = batch['images'].to(self.device)
                condition = batch.get('condition')
                if condition is not None:
                    condition = condition.to(self.device)
            else:
                # 普通张量批次
                x = batch[0].to(self.device)
                condition = None
             
            # 前向传播
            self.optimizer.zero_grad()
            loss = self.model(x, condition=condition)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get("gradient_clip", 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get("gradient_clip", 1.0)
                )
                
            # 参数更新
            self.optimizer.step()
            
            # 更新EMA模型
            if self.use_ema:
                self.ema.update()
                
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 更新进度条
            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({"loss": current_loss})
            
            # 更新步数
            self.step += 1
            
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        self.losses.append(avg_loss)
        
        # 打印信息
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            logger.warning("No validation dataloader provided, skipping validation.")
            return None
            
        self.model.eval()
        
        # 如果使用EMA，应用EMA参数
        if self.use_ema:
            self.ema.apply_shadow()
            
        val_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            # 将数据移动到设备
            if isinstance(batch, dict):
                x = batch['images'].to(self.device)
                condition = batch.get('condition')
                if condition is not None:
                    condition = condition.to(self.device)
            else:
                x = batch[0].to(self.device)
                condition = None
                
            # 计算验证损失
            loss = self.model(x, condition=condition)
            val_loss += loss.item()
            
            pbar.set_postfix({"loss": loss.item()})
            
        # 恢复原始模型参数
        if self.use_ema:
            self.ema.restore()
            
        avg_val_loss = val_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        
        return avg_val_loss
    
    def train(self, num_epochs=100):
        """
        完整训练流程
        
        参数:
            num_epochs: 训练的周期数
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            # 训练一个周期
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 保存检查点
            is_best = False
            if val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                is_best = True
                
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(checkpoint_path, is_best)
            
        logger.info("Training completed!") 