#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
扩散模型图像生成脚本

用于从训练好的扩散模型生成图像，支持不同的采样方法和条件生成
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import logging
import sys
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from inference.pipeline import DiffusionPipeline
from inference.sampling import DDPMSampler, DDIMSampler, PNDMSampler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="扩散模型图像生成")
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='生成图像的大小')
    
    # 采样参数
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim', 'pndm'],
                        help='采样器类型')
    parser.add_argument('--steps', type=int, default=100,
                        help='采样步数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='生成批次大小')
    parser.add_argument('--num_images', type=int, default=4,
                        help='生成图像数量')
    
    # 条件参数(可选)
    parser.add_argument('--condition_image', type=str, default=None,
                        help='条件图像路径(用于条件生成)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备(cuda或cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子(用于可重复结果)')
    
    return parser.parse_args()


def main():
    """主生成流程"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子(如果提供)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建生成管道
    logger.info(f"加载模型: {args.model_path}")
    pipeline = DiffusionPipeline(
        model_path=args.model_path,
        device=device,
        sampler_type=args.sampler,
        sampling_steps=args.steps,
        image_size=args.image_size
    )
    
    # 处理条件(如果提供)
    condition = None
    if args.condition_image:
        logger.info(f"使用条件图像: {args.condition_image}")
        condition = pipeline.condition_image(args.condition_image)
    
    # 生成图像
    logger.info(f"开始生成 {args.num_images} 张图像，使用 {args.sampler} 采样器...")
    
    # 计算需要多少批次
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    total_images = 0
    
    for batch_idx in range(num_batches):
        # 计算当前批次大小
        current_batch_size = min(args.batch_size, args.num_images - total_images)
        
        # 生成图像
        logger.info(f"生成批次 {batch_idx+1}/{num_batches}，批次大小: {current_batch_size}")
        images = pipeline(
            batch_size=current_batch_size,
            condition=condition,
            seed=args.seed+batch_idx if args.seed is not None else None,
            return_pil=True
        )
        
        # 保存图像
        for i, img in enumerate(images):
            img_path = os.path.join(args.output_dir, f"generated_{total_images+i:04d}.png")
            img.save(img_path)
            logger.info(f"保存图像到: {img_path}")
            
        # 更新计数
        total_images += len(images)
    
    logger.info(f"完成生成 {total_images} 张图像!")


if __name__ == "__main__":
    main() 