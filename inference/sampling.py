"""
扩散模型采样模块

提供了不同的扩散模型采样算法:
- DDPM (去噪扩散概率模型采样)
- DDIM (去噪扩散隐式模型采样)
- 加速采样方法
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Callable


class DiffusionSampler:
    """
    扩散模型采样基类
    """
    def __init__(self, model, diffusion_steps=1000, device=None):
        """
        初始化采样器
        
        参数:
            model: 扩散模型实例
            diffusion_steps: 扩散步数
            device: 计算设备
        """
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    @torch.no_grad()
    def sample(self, 
               batch_size: int = 1, 
               image_size: Tuple[int, int, int] = (3, 256, 256),
               condition = None,
               clip_denoised: bool = True,
               verbose: bool = True):
        """
        生成样本(在子类中实现)
        
        参数:
            batch_size: 批次大小
            image_size: 图像大小(通道, 高度, 宽度)
            condition: 条件输入(如果是条件扩散模型)
            clip_denoised: 是否裁剪去噪后的图像到[-1, 1]
            verbose: 是否显示进度条
            
        返回:
            生成的样本
        """
        raise NotImplementedError("在子类中实现")
        

class DDPMSampler(DiffusionSampler):
    """
    DDPM (去噪扩散概率模型) 采样器
    这是原始的逐步采样方法
    """
    
    @torch.no_grad()
    def sample(self, 
               batch_size: int = 1, 
               image_size: Tuple[int, int, int] = (3, 256, 256),
               condition = None,
               clip_denoised: bool = True,
               verbose: bool = True):
        """
        使用DDPM采样生成图像
        
        参数:
            batch_size: 批次大小
            image_size: 图像大小(通道, 高度, 宽度)
            condition: 条件输入(如果是条件扩散模型)
            clip_denoised: 是否裁剪去噪后的图像到[-1, 1]
            verbose: 是否显示进度条
            
        返回:
            生成的图像
        """
        # 从噪声开始
        x = torch.randn(batch_size, *image_size, device=self.device)
        
        # 移动条件(如果有)到设备
        if condition is not None and isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
        
        # 迭代
        iterator = range(self.diffusion_steps - 1, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc="DDPM Sampling")
            
        # 逐步去噪
        for i in iterator:
            # 当前时间步
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            
            # 预测噪声
            predicted_noise = self.model(x, t, condition)
            
            # 获取当前步骤的参数
            alpha = self.model.alphas[i]
            alpha_cumprod = self.model.alphas_cumprod[i]
            beta = self.model.betas[i]
            
            # 添加噪声或不添加(最后一步)
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # 去噪步骤
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            # 可选的裁剪
            if clip_denoised and i == 0:
                x = torch.clamp(x, -1.0, 1.0)
                
        return x


class DDIMSampler(DiffusionSampler):
    """
    DDIM (去噪扩散隐式模型) 采样器
    这是一种加速采样方法，可以使用更少的步骤
    """
    
    def __init__(self, model, diffusion_steps=1000, sampling_steps=100, device=None, eta=0.0):
        """
        初始化DDIM采样器
        
        参数:
            model: 扩散模型
            diffusion_steps: 训练中使用的扩散步数
            sampling_steps: 采样时使用的步数(少于训练步数)
            device: 计算设备
            eta: DDIM的噪声系数(0=确定性)
        """
        super().__init__(model, diffusion_steps, device)
        self.sampling_steps = sampling_steps
        self.eta = eta
        
        # 计算DDIM时间步
        c = self.diffusion_steps // self.sampling_steps
        self.timesteps = np.asarray(list(range(0, self.diffusion_steps, c)))
        
    @torch.no_grad()
    def sample(self, 
               batch_size: int = 1, 
               image_size: Tuple[int, int, int] = (3, 256, 256),
               condition = None,
               clip_denoised: bool = True,
               verbose: bool = True):
        """
        使用DDIM采样生成图像
        
        参数:
            batch_size: 批次大小
            image_size: 图像大小(通道, 高度, 宽度)
            condition: 条件输入(如果是条件扩散模型)
            clip_denoised: 是否裁剪去噪后的图像到[-1, 1]
            verbose: 是否显示进度条
            
        返回:
            生成的图像
        """
        # 从噪声开始
        x = torch.randn(batch_size, *image_size, device=self.device)
        
        # 移动条件(如果有)到设备
        if condition is not None and isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
        
        # 迭代
        iterator = range(len(self.timesteps) - 1, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc="DDIM Sampling")
            
        # 采样步骤
        for i in iterator:
            # 当前和前一时间步
            t_cur = self.timesteps[i]
            t_prev = 0 if i == 0 else self.timesteps[i-1]
            
            # 当前时间索引
            t = torch.full((batch_size,), t_cur, dtype=torch.long, device=self.device)
            
            # 预测噪声
            predicted_noise = self.model(x, t, condition)
            
            # 获取参数
            alpha_cumprod_t = self.model.alphas_cumprod[t_cur]
            alpha_cumprod_t_prev = self.model.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha_cumprod_t)
            
            # 预测x_0
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # 裁剪预测的x_0
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
                
            # DDIM采样公式的参数
            eta = self.eta
            
            # 计算DDIM去噪方向
            sqrt_one_minus_at = torch.sqrt(1 - alpha_cumprod_t)
            sqrt_one_minus_at_prev = torch.sqrt(1 - alpha_cumprod_t_prev)
            
            # 确定部分
            c1 = torch.sqrt(alpha_cumprod_t_prev) * sqrt_one_minus_at / sqrt_one_minus_at_prev * torch.sqrt(1 - alpha_cumprod_t_prev / alpha_cumprod_t)
            
            # 随机部分(eta=0时为0)
            c2 = sqrt_one_minus_at_prev * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev) * eta
            
            # 更新x
            x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + c1 * predicted_noise + c2 * torch.randn_like(x)
            
        return x


class PredictionTypeEnum:
    """预测类型枚举"""
    EPSILON = "epsilon"  # 直接预测噪声
    X_START = "x_start"  # 预测原始图像
    V_PREDICTION = "v_prediction"  # 预测v


class PNDMSampler(DiffusionSampler):
    """
    PNDM (伪数值扩散模型) 采样器
    使用高阶求解器来加速采样
    """
    
    def __init__(self, model, diffusion_steps=1000, sampling_steps=50, device=None, 
                 prediction_type=PredictionTypeEnum.EPSILON):
        """
        初始化PNDM采样器
        
        参数:
            model: 扩散模型
            diffusion_steps: 训练中使用的扩散步数
            sampling_steps: 采样时使用的步数
            device: 计算设备
            prediction_type: 模型预测的类型
        """
        super().__init__(model, diffusion_steps, device)
        self.sampling_steps = sampling_steps
        self.prediction_type = prediction_type
        
        # 计算采样时间步
        c = self.diffusion_steps // self.sampling_steps
        self.timesteps = np.asarray(list(range(0, self.diffusion_steps, c)))
        
    @torch.no_grad()
    def sample(self, 
               batch_size: int = 1, 
               image_size: Tuple[int, int, int] = (3, 256, 256),
               condition = None,
               clip_denoised: bool = True,
               verbose: bool = True):
        """
        使用PNDM采样生成图像
        
        参数:
            batch_size: 批次大小
            image_size: 图像大小(通道, 高度, 宽度)
            condition: 条件输入(如果是条件扩散模型)
            clip_denoised: 是否裁剪去噪后的图像到[-1, 1]
            verbose: 是否显示进度条
            
        返回:
            生成的图像
        """
        # 从噪声开始
        x = torch.randn(batch_size, *image_size, device=self.device)
        
        # 移动条件(如果有)到设备
        if condition is not None and isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
            
        # 对于PNDM，我们使用4阶Runge-Kutta方法
        # 需要保存中间导数
        d1 = d2 = d3 = d4 = None
        x_prev = x_prev_prev = None
        
        # 迭代
        iterator = range(0, len(self.timesteps))
        if verbose:
            iterator = tqdm(iterator, desc="PNDM Sampling")
            
        for i in iterator:
            # 当前时间步
            t_index = len(self.timesteps) - 1 - i
            t = torch.full((batch_size,), self.timesteps[t_index], dtype=torch.long, device=self.device)
            
            # 当处于前几步时，使用简单的DDIM步骤
            if i < 4:
                # 预测噪声
                pred = self.model(x, t, condition)
                
                # 获取参数
                alpha = self.model.alphas_cumprod[t]
                alpha_prev = self.model.alphas_cumprod[t-1] if t_index > 0 else torch.ones_like(alpha)
                
                # DDIM步骤
                if self.prediction_type == PredictionTypeEnum.EPSILON:
                    # 直接预测噪声的情况
                    x0_pred = (x - torch.sqrt(1 - alpha) * pred) / torch.sqrt(alpha)
                elif self.prediction_type == PredictionTypeEnum.X_START:
                    # 预测原始图像的情况
                    x0_pred = pred
                else:
                    # 预测v的情况
                    x0_pred = torch.sqrt(alpha) * x - torch.sqrt(1 - alpha) * pred
                    
                # 裁剪
                if clip_denoised:
                    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
                    
                # 计算噪声分量
                noise = torch.randn_like(x) if t_index > 0 else torch.zeros_like(x)
                
                # 下一步
                x_next = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise
                
                # 保存中间变量
                if i == 0:
                    x_prev = x_next
                elif i == 1:
                    x_prev_prev = x_prev
                    x_prev = x_next
                elif i == 2:
                    d1 = x_next - x_prev
                    d2 = x_prev - x_prev_prev
                    x_prev_prev = x_prev
                    x_prev = x_next
                elif i == 3:
                    d3 = x_next - x_prev
                    d4 = d3 - d2
                    x_prev = x_next
                    
                x = x_next
                
            else:
                # 使用PNDM的高阶更新
                # 使用之前保存的导数进行Adams-Bashforth更新
                x_next = x + (55/24) * d1 - (59/24) * d2 + (37/24) * d3 - (9/24) * d4
                
                # 更新导数
                d4 = d3
                d3 = d2
                d2 = d1
                d1 = x_next - x
                
                x = x_next
                
        # 最终裁剪
        if clip_denoised:
            x = torch.clamp(x, -1.0, 1.0)
            
        return x 