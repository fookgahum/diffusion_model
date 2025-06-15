#!/usr/bin/env python3
"""
扩散模型数据集最终准备脚本
提供多种数据源选择，确保训练能够顺利进行
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from PIL import Image

class CustomImageDataset(Dataset):
    """自定义图像数据集"""
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if isinstance(self.images, torch.Tensor):
            image = self.images[idx]
        else:
            image = Image.fromarray(self.images[idx])
            if self.transform:
                image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image

def create_synthetic_dataset(train_size=5000, test_size=1000):
    """创建合成数据集用于扩散模型训练"""
    print(f"创建合成数据集: 训练集 {train_size} 样本, 测试集 {test_size} 样本")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 生成有结构的合成图像
    def generate_structured_images(size, image_shape=(32, 32, 3)):
        images = []
        for i in range(size):
            # 创建不同类型的图像模式
            pattern = i % 4
            img = np.zeros(image_shape, dtype=np.uint8)
            
            if pattern == 0:  # 渐变图像
                for y in range(image_shape[0]):
                    for x in range(image_shape[1]):
                        img[y, x] = [x * 255 // image_shape[1], 
                                   y * 255 // image_shape[0], 
                                   (x + y) * 255 // (image_shape[0] + image_shape[1])]
            elif pattern == 1:  # 几何图形
                center = (image_shape[0] // 2, image_shape[1] // 2)
                for y in range(image_shape[0]):
                    for x in range(image_shape[1]):
                        dist = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                        if dist < image_shape[0] // 3:
                            img[y, x] = [255, 100, 50]
            elif pattern == 2:  # 条纹图案
                for y in range(image_shape[0]):
                    if (y // 4) % 2:
                        img[y, :] = [255, 255, 0]
                    else:
                        img[y, :] = [0, 255, 255]
            else:  # 随机噪声
                img = np.random.randint(0, 256, image_shape, dtype=np.uint8)
            
            images.append(img)
        
        return np.array(images)
    
    # 生成训练和测试图像
    train_images = generate_structured_images(train_size)
    test_images = generate_structured_images(test_size)
    
    # 创建标签 (对扩散模型不是必需的，但保留用于可能的条件生成)
    train_labels = torch.randint(0, 4, (train_size,))
    test_labels = torch.randint(0, 4, (test_size,))
    
    # 创建数据集
    train_dataset = CustomImageDataset(train_images, train_labels, transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform)
    
    return train_dataset, test_dataset

def create_minimal_cifar_like_dataset():
    """创建最小的CIFAR风格数据集"""
    print("创建最小CIFAR风格数据集...")
    
    # 直接创建tensor数据集（更快）
    train_images = torch.randn(2000, 3, 32, 32)  # 2000个训练样本
    test_images = torch.randn(400, 3, 32, 32)    # 400个测试样本
    
    # 归一化到[-1, 1]
    train_images = torch.tanh(train_images)
    test_images = torch.tanh(test_images)
    
    train_dataset = TensorDataset(train_images)
    test_dataset = TensorDataset(test_images)
    
    return train_dataset, test_dataset

def test_dataset_loading(dataset, name="数据集", batch_size=8, num_batches=3):
    """测试数据集加载"""
    print(f"\n测试 {name}:")
    print(f"- 样本总数: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_batches = len(dataloader)
    print(f"- 批次总数: {total_batches}")
    
    # 测试前几个批次
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        if isinstance(batch, tuple):
            images, labels = batch
            print(f"- 批次 {i+1}: 图像 {images.shape}, 标签 {labels.shape}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        else:
            images = batch[0] if isinstance(batch, list) else batch
            print(f"- 批次 {i+1}: 图像 {images.shape}")
            print(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
    
    print(f"✅ {name} 测试通过")

def save_dataset_info(train_dataset, test_dataset, data_dir):
    """保存数据集信息"""
    info = {
        '数据集类型': '合成扩散训练数据',
        '训练样本数': len(train_dataset),
        '测试样本数': len(test_dataset),
        '图像尺寸': '32x32x3',
        '数据范围': '[-1, 1]',
        '用途': '扩散模型训练和Jailbreak研究'
    }
    
    info_file = Path(data_dir) / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("扩散模型训练数据集信息\n")
        f.write("=" * 40 + "\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("使用说明:\n")
        f.write("1. 此数据集专为扩散模型训练设计\n")
        f.write("2. 图像已归一化到[-1,1]范围\n")
        f.write("3. 适合DDPM/DDIM等扩散模型架构\n")
        f.write("4. 可用于后续的jailbreak攻击研究\n")
    
    print(f"数据集信息已保存到: {info_file}")

def main():
    print("=" * 60)
    print("扩散模型数据集最终准备工具")
    print("=" * 60)
    
    # 创建目录
    data_dir = Path("./data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n选择数据集类型:")
    print("1. 结构化合成数据集 (推荐用于训练)")
    print("2. 最小张量数据集 (快速测试)")
    
    # 为了自动化，我们选择选项1
    choice = "1"
    
    if choice == "1":
        train_dataset, test_dataset = create_synthetic_dataset(
            train_size=5000, test_size=1000
        )
        dataset_type = "结构化合成数据集"
    else:
        train_dataset, test_dataset = create_minimal_cifar_like_dataset()
        dataset_type = "最小张量数据集"
    
    print(f"\n使用 {dataset_type}")
    
    # 测试数据集
    test_dataset_loading(train_dataset, "训练集")
    test_dataset_loading(test_dataset, "测试集")
    
    # 保存数据集信息
    save_dataset_info(train_dataset, test_dataset, data_dir)
    
    print("\n" + "=" * 60)
    print("✅ 数据集准备完成！")
    print("✅ 现在可以开始训练扩散模型了")
    print("✅ 训练完成后即可进行jailbreak研究")
    print("=" * 60)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = main() 