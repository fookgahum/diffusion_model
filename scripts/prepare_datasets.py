#!/usr/bin/env python3
"""
数据集准备脚本
支持下载和准备多种扩散模型训练数据集
"""

import os
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def download_file(url, filepath):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def prepare_cifar10(data_dir):
    """准备CIFAR-10数据集"""
    print("准备CIFAR-10数据集...")
    
    # 下载CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    
    print(f"CIFAR-10准备完成: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    return train_dataset, test_dataset


def prepare_celeba(data_dir):
    """准备CelebA数据集"""
    print("准备CelebA数据集...")
    print("注意: CelebA需要手动下载，请访问 http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    
    celeba_dir = Path(data_dir) / "celeba"
    celeba_dir.mkdir(exist_ok=True)
    
    # 检查是否已存在
    img_dir = celeba_dir / "img_align_celeba"
    if img_dir.exists():
        print(f"发现CelebA数据: {img_dir}")
        return True
    else:
        print("请手动下载CelebA数据集并解压到:", celeba_dir)
        return False


def prepare_coco(data_dir):
    """准备MS-COCO数据集"""
    print("准备MS-COCO数据集...")
    coco_dir = Path(data_dir) / "coco"
    coco_dir.mkdir(exist_ok=True)
    
    # COCO URLs
    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    for filename, url in urls.items():
        filepath = coco_dir / filename
        if not filepath.exists():
            print(f"下载 {filename}...")
            download_file(url, filepath)
            
            # 解压
            print(f"解压 {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
        else:
            print(f"{filename} 已存在，跳过下载")
    
    print("MS-COCO数据集准备完成")


def create_custom_dataset_template(data_dir):
    """创建自定义数据集模板"""
    custom_dir = Path(data_dir) / "custom"
    custom_dir.mkdir(exist_ok=True)
    
    train_dir = custom_dir / "train"
    val_dir = custom_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 创建README
    readme_content = """
# 自定义数据集使用说明

## 目录结构
```
custom/
├── train/          # 训练图片
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── val/            # 验证图片
│   ├── image1.jpg
│   └── ...
└── README.md       # 本文件
```

## 使用方法
1. 将训练图片放入 train/ 目录
2. 将验证图片放入 val/ 目录
3. 支持的格式: .jpg, .jpeg, .png, .bmp
4. 建议图片分辨率: 256x256 或更高

## 文本条件数据集
如果需要文本条件，请创建对应的 .txt 文件：
- image1.jpg -> image1.txt (包含图片描述)
"""
    
    with open(custom_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"自定义数据集模板已创建: {custom_dir}")


def main():
    parser = argparse.ArgumentParser(description="准备扩散模型训练数据集")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "celeba", "coco", "custom"], 
                       default="cifar10", help="选择要准备的数据集")
    parser.add_argument("--data_dir", type=str, default="./data/raw", 
                       help="数据存储目录")
    
    args = parser.parse_args()
    
    # 创建目录
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"数据目录: {data_dir.absolute()}")
    
    if args.dataset == "cifar10":
        prepare_cifar10(data_dir)
    elif args.dataset == "celeba":
        prepare_celeba(data_dir)
    elif args.dataset == "coco":
        prepare_coco(data_dir)
    elif args.dataset == "custom":
        create_custom_dataset_template(data_dir)
    
    print("数据集准备完成！")


if __name__ == "__main__":
    main() 