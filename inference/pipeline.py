"""
扩散模型端到端推理管道

提供了完整的推理流程，从模型加载到图像生成
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

# 从同级目录导入采样器
from .sampling import DiffusionSampler, DDPMSampler, DDIMSampler, PNDMSampler


class DiffusionPipeline:
    """
    扩散模型推理管道
    封装了模型加载和推理过程
    """
    
    def __init__(self, 
                 model_path: str,       # 模型路径
                 device = None,         # 计算设备
                 sampler_type: str = "ddim",  # 采样器类型
                 sampling_steps: int = 100,   # 采样步数
                 image_size: int = 256,       # 输出图像大小
                 ):
        """
        初始化推理管道
        
        参数:
            model_path: 模型检查点路径
            device: 计算设备(如果为None则自动选择)
            sampler_type: 采样器类型('ddpm', 'ddim', 'pndm')
            sampling_steps: 采样步数
            image_size: 输出图像大小
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # 加载模型
        self._load_model()
        
        # 采样器类型和步数
        self.sampler_type = sampler_type.lower()
        self.sampling_steps = sampling_steps
        
        # 创建采样器
        self._create_sampler()
        
        # 转换模块
        self.to_pil = transforms.ToPILImage()
        
    def _load_model(self):
        """加载模型"""
        print(f"Loading model from {self.model_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 检查点中应该包含模型架构和权重信息
        # 这里假设检查点中有完整的模型定义，实际情况可能需要先定义模型架构
        if 'model' in checkpoint:
            # state_dict方式加载
            self.model_config = checkpoint.get('config', {})
            
            # 这里需要导入具体的模型定义
            # 例如 from models.diffusion.ddpm import DiffusionModel
            # 由于示例中没有完整的模型导入，这里只是占位
            # 使用时请替换为实际的模型类和加载方式
            print("加载模型: 在实际使用中请替换为实际的模型加载代码")
            print("例如: self.model = DiffusionModel(...)")
            print("self.model.load_state_dict(checkpoint['model'])")
            
            # 模拟代码 (需要在实际使用中更改)
            print("警告: 当前使用模拟代码，实际使用时请更改为真实的模型加载代码")
            from models.diffusion.ddpm import DiffusionModel
            from models.diffusion.unet import UNet
            
            # 重建UNet和扩散模型
            unet = UNet()
            self.model = DiffusionModel(unet, self.model_config)
            
            # 加载状态字典
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise ValueError(f"Checkpoint does not contain expected model information")
        
        # 移动模型到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully to {self.device}")
        
    def _create_sampler(self):
        """创建采样器"""
        if self.sampler_type == "ddpm":
            self.sampler = DDPMSampler(
                model=self.model,
                diffusion_steps=self.model.config.timesteps,
                device=self.device
            )
        elif self.sampler_type == "ddim":
            self.sampler = DDIMSampler(
                model=self.model,
                diffusion_steps=self.model.config.timesteps,
                sampling_steps=self.sampling_steps,
                device=self.device,
                eta=0.0  # 设置为0表示完全确定性的DDIM
            )
        elif self.sampler_type == "pndm":
            self.sampler = PNDMSampler(
                model=self.model,
                diffusion_steps=self.model.config.timesteps,
                sampling_steps=self.sampling_steps,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}, must be one of ['ddpm', 'ddim', 'pndm']")
            
    def __call__(self, 
                 batch_size: int = 1,
                 condition = None,
                 seed: Optional[int] = None,
                 return_pil: bool = True):
        """
        生成图像
        
        参数:
            batch_size: 生成图像的数量
            condition: 条件输入(如果是条件模型)
            seed: 随机种子(用于可复现结果)
            return_pil: 是否返回PIL图像
            
        返回:
            生成的图像列表
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # 图像形状
        image_shape = (3, self.image_size, self.image_size)
        
        # 使用采样器生成图像
        samples = self.sampler.sample(
            batch_size=batch_size,
            image_size=image_shape,
            condition=condition,
            clip_denoised=True,
            verbose=True
        )
        
        # 转换为PIL图像
        if return_pil:
            # 归一化到0-1范围(从[-1, 1]范围)
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            
            # 转换为PIL图像
            images = []
            for i in range(samples.shape[0]):
                img = self.to_pil(samples[i])
                images.append(img)
                
            return images
        
        # 返回张量
        return samples
        
    def condition_image(self, image_path: str):
        """
        处理条件图像(用于条件生成)
        
        参数:
            image_path: 图像路径
            
        返回:
            处理后的条件张量
        """
        # 图像转换
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 加载并转换图像
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        return image.to(self.device)
    
    def save_images(self, images: List[Image.Image], output_dir: str = "outputs"):
        """
        保存生成的图像
        
        参数:
            images: PIL图像列表
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个图像
        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f"generated_{i:04d}.png")
            img.save(img_path)
            print(f"Saved image to {img_path}")


class TextToImagePipeline(DiffusionPipeline):
    """
    文本到图像的扩散模型推理管道
    """
    
    def __init__(self, 
                 model_path: str,        # 模型路径
                 tokenizer,              # 分词器
                 device = None,          # 计算设备
                 sampler_type: str = "ddim",  # 采样器类型
                 sampling_steps: int = 100,   # 采样步数
                 image_size: int = 256,       # 输出图像大小
                 max_text_length: int = 77,   # 最大文本长度
                 ):
        """
        初始化文本到图像推理管道
        
        参数:
            model_path: 模型检查点路径
            tokenizer: 文本分词器
            device: 计算设备
            sampler_type: 采样器类型
            sampling_steps: 采样步数
            image_size: 输出图像大小
            max_text_length: 最大文本长度
        """
        super().__init__(
            model_path=model_path, 
            device=device,
            sampler_type=sampler_type,
            sampling_steps=sampling_steps,
            image_size=image_size
        )
        
        # 设置分词器
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
    def condition_text(self, text: str):
        """
        处理文本条件
        
        参数:
            text: 文本描述
            
        返回:
            处理后的文本令牌
        """
        # 使用tokenizer处理文本
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        for k, v in tokens.items():
            tokens[k] = v.to(self.device)
            
        return tokens
    
    def __call__(self, 
                 prompt: str,
                 batch_size: int = 1,
                 seed: Optional[int] = None,
                 return_pil: bool = True):
        """
        根据文本提示生成图像
        
        参数:
            prompt: 文本提示
            batch_size: 生成图像的数量
            seed: 随机种子
            return_pil: 是否返回PIL图像
            
        返回:
            生成的图像列表
        """
        # 处理文本条件
        tokens = self.condition_text(prompt)
        
        # 调用父类方法生成图像
        return super().__call__(
            batch_size=batch_size,
            condition=tokens,
            seed=seed,
            return_pil=return_pil
        ) 