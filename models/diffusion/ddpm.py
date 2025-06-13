"""
扩散概率模型(Denoising Diffusion Probabilistic Models)的实现

参考文献:
- Ho, J., Jain, A., & Abbeel, P. (2020). 
  Denoising diffusion probabilistic models. 
  Advances in Neural Information Processing Systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class DiffusionModel(nn.Module):
    """
    扩散概率模型的基类
    实现了前向扩散过程和反向采样过程
    """
    
    def __init__(self, noise_predictor, config):
        """
        初始化扩散模型
        
        参数:
            noise_predictor: 噪声预测模型，通常是UNet
            config: 配置对象，包含扩散过程的参数
        """
        super().__init__()
        self.noise_predictor = noise_predictor
        self.config = config
        
        # 设置扩散过程的beta值调度
        self.setup_noise_schedule()
        
        # 初始化用于加速计算的参数
        self.setup_sampling_parameters()
        
    def setup_noise_schedule(self):
        """设置噪声调度参数"""
        # 根据配置的beta调度类型设置beta值序列
        if self.config.beta_schedule == 'linear':
            self.betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.timesteps
            )
        elif self.config.beta_schedule == 'cosine':
            # 余弦调度
            steps = self.config.timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.config.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.config.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"未知的beta调度类型: {self.config.beta_schedule}")
            
        # 计算alpha值和累积值
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算其他用于采样的辅助值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def setup_sampling_parameters(self):
        """设置采样过程中需要的参数"""
        # 为DDIM采样准备参数
        if self.config.use_ddim:
            c = self.config.timesteps // self.config.sampling_steps
            self.ddim_timesteps = np.asarray(list(range(0, self.config.timesteps, c)))
            self.ddim_timesteps_prev = np.append([-1], self.ddim_timesteps[:-1])
    
    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散过程: q(x_t | x_0)
        
        参数:
            x_0: 原始数据
            t: 时间步
            noise: 噪声，如果为None则随机生成
            
        返回:
            x_t: 在t时刻的噪声数据
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # 计算x_t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def predict_noise(self, x_t, t, condition=None):
        """
        预测噪声
        
        参数:
            x_t: 输入的噪声数据
            t: 时间步
            condition: 条件输入(如果有)
            
        返回:
            预测的噪声
        """
        return self.noise_predictor(x_t, t, condition)
    
    def forward(self, x_0, condition=None):
        """
        训练过程的前向传播
        
        参数:
            x_0: 原始输入数据
            condition: 条件输入(如果有)
            
        返回:
            损失值
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 随机选择时间步
        t = torch.randint(0, self.config.timesteps, (batch_size,), device=device)
        
        # 添加噪声
        x_t, noise = self.q_sample(x_0, t)
        
        # 预测噪声
        predicted_noise = self.predict_noise(x_t, t, condition)
        
        # 计算简单MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
        
    @torch.no_grad()
    def sample(self, batch_size, img_shape, device, condition=None):
        """
        从噪声生成样本
        
        参数:
            batch_size: 批次大小
            img_shape: 图像形状
            device: 设备
            condition: 条件输入(如果有)
            
        返回:
            生成的样本
        """
        # 从纯噪声开始
        x = torch.randn(batch_size, *img_shape, device=device)
        
        # 选择采样方法
        if self.config.use_ddim:
            return self._sample_ddim(x, condition)
        else:
            return self._sample_ddpm(x, condition)
    
    @torch.no_grad()
    def _sample_ddpm(self, x, condition=None):
        """标准DDPM采样过程"""
        for i in range(self.config.timesteps - 1, -1, -1):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.predict_noise(x, t, condition)
            
            # 计算均值
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # 去噪步骤
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            # 可选的裁剪
            if self.config.clip_denoised and i == 0:
                x = torch.clamp(x, -1, 1)
                
        return x
    
    @torch.no_grad()
    def _sample_ddim(self, x, condition=None):
        """DDIM采样过程"""
        for i in range(len(self.ddim_timesteps) - 1, -1, -1):
            t_now = self.ddim_timesteps[i]
            t_prev = self.ddim_timesteps_prev[i]
            
            # 当前时间步
            t = torch.full((x.shape[0],), t_now, device=x.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.predict_noise(x, t, condition)
            
            # 当前alpha参数
            alpha_cumprod_t = self.alphas_cumprod[t_now]
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha_cumprod_t)
            
            # DDIM采样公式
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if self.config.clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1, 1)
                
            # 方差
            eta = self.config.ddim_sampling_eta
            
            # 计算"方向"向量
            direction = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * predicted_noise
            
            # 计算噪声分量
            noise = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev) * predicted_noise
            
            # 更新x
            x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + direction + noise
            
        return x 