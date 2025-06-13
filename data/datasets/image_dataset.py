"""
图像数据集模块

提供用于加载和预处理扩散模型训练数据的工具
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Union, Callable


class DiffusionDataset(Dataset):
    """
    扩散模型图像数据集基类
    """
    def __init__(self, 
                 image_paths: List[str],           # 图像路径列表
                 transform=None,                   # 图像转换
                 condition_paths: List[str] = None, # 条件图像路径(如果有)
                 condition_transform=None,         # 条件转换
                 image_size: int = 256,            # 图像大小
                 ):
        """
        初始化数据集
        
        参数:
            image_paths: 图像文件路径列表
            transform: 图像转换函数
            condition_paths: 条件图像路径列表(如果是条件模型)
            condition_transform: 条件图像的转换函数
            image_size: 输出图像的大小
        """
        self.image_paths = image_paths
        self.condition_paths = condition_paths
        self.is_conditional = condition_paths is not None
        
        # 默认转换(如果未提供)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
            
        # 条件转换
        if self.is_conditional and condition_transform is None:
            self.condition_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            ])
        else:
            self.condition_transform = condition_transform
            
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        # 处理条件(如果有)
        if self.is_conditional:
            cond_path = self.condition_paths[idx]
            condition = Image.open(cond_path).convert('RGB')
            
            if self.condition_transform:
                condition = self.condition_transform(condition)
                
            return {'images': image, 'condition': condition}
        else:
            return image


class ImageFolderDataset(DiffusionDataset):
    """
    从文件夹加载图像的数据集
    """
    def __init__(self, 
                 root_dir: str,              # 图像根目录
                 transform=None,             # 图像转换
                 condition_dir: str = None,  # 条件图像目录
                 condition_transform=None,   # 条件转换
                 image_size: int = 256,      # 图像大小
                 extensions: List[str] = ['.jpg', '.jpeg', '.png'],  # 支持的扩展名
                 ):
        """
        初始化文件夹数据集
        
        参数:
            root_dir: 图像根目录
            transform: 图像转换函数
            condition_dir: 条件图像目录
            condition_transform: 条件图像转换函数
            image_size: 输出图像大小
            extensions: 支持的文件扩展名列表
        """
        # 收集图像路径
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
                    
        # 收集条件路径(如果有)
        condition_paths = None
        if condition_dir:
            condition_paths = []
            for img_path in image_paths:
                # 从图像路径构建条件路径
                # 这里的实现假设条件图像与原始图像同名，可以根据需要修改
                rel_path = os.path.relpath(img_path, root_dir)
                cond_path = os.path.join(condition_dir, rel_path)
                
                if os.path.exists(cond_path):
                    condition_paths.append(cond_path)
                else:
                    # 如果找不到对应条件，跳过这一对
                    image_paths.remove(img_path)
                    
        # 初始化基类
        super().__init__(
            image_paths=image_paths,
            transform=transform,
            condition_paths=condition_paths,
            condition_transform=condition_transform,
            image_size=image_size
        )


class TextConditionedDataset(Dataset):
    """
    文本条件的图像数据集
    用于文本到图像生成任务
    """
    def __init__(self, 
                 image_paths: List[str],       # 图像路径列表
                 captions: List[str],          # 对应的文本标题
                 tokenizer,                    # 文本标记器
                 transform=None,               # 图像转换
                 max_text_length: int = 77,    # 最大文本长度
                 image_size: int = 256,        # 图像大小
                 ):
        """
        初始化文本条件数据集
        
        参数:
            image_paths: 图像文件路径列表
            captions: 对应的文本描述
            tokenizer: 用于处理文本的分词器
            transform: 图像转换函数
            max_text_length: 最大文本长度
            image_size: 输出图像大小
        """
        assert len(image_paths) == len(captions), "图像和文本标题数量必须相同"
        
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # 默认转换(如果未提供)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        # 处理文本
        caption = self.captions[idx]
        
        # 使用tokenizer处理文本
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # 移除批次维度
        for k, v in tokens.items():
            tokens[k] = v.squeeze(0)
            
        return {'images': image, 'tokens': tokens}


def get_dataloaders(dataset, batch_size=32, val_split=0.1, num_workers=4, seed=42):
    """
    从数据集创建训练和验证数据加载器
    
    参数:
        dataset: 数据集实例
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
        seed: 随机种子
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 设置随机种子
    torch.manual_seed(seed)
    
    # 数据集大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 随机打乱
    np.random.shuffle(indices)
    
    # 分割点
    val_size = int(np.floor(val_split * dataset_size))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 创建采样器
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 