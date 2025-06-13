"""
UNet模型实现

扩散模型中的骨干网络，用于预测噪声
包含下采样、上采样和残差块结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码，用于时间步编码
    """
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_positions) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 处理奇数维度情况
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings


class Block(nn.Module):
    """
    U-Net的基本块
    包含两个卷积层和残差连接
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        
        # 主要卷积层
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # 残差连接
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, time_emb=None):
        # 残差路径
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # 加入时间嵌入
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(F.silu(time_emb))
            h = h + time_emb[..., None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # 残差连接
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    自注意力模块
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        
        # 分离Q,K,V并转换形状
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 形状: [b, heads, c', h*w]
        
        # 计算注意力得分
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        
        return self.proj(out)


class UNet(nn.Module):
    """
    扩散模型中的UNet架构
    """
    def __init__(self, 
                 in_channels=3,        # 输入通道数
                 model_channels=128,   # 基础通道数
                 out_channels=3,       # 输出通道数
                 num_res_blocks=2,     # 每个尺度的残差块数量
                 attention_resolutions=(8, 4),  # 使用注意力的分辨率
                 channel_mult=(1, 2, 4, 8),     # 通道乘数
                 time_emb_dim=512):    # 时间嵌入维度
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 输入卷积
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 下采样部分
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = model_channels
        cur_res = 1  # 开始的分辨率
        
        # 构建下采样层
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # 添加残差块
                self.down_blocks.append(
                    Block(channels, model_channels * mult, time_emb_dim=time_emb_dim)
                )
                channels = model_channels * mult
                
                # 如果当前分辨率需要添加注意力层
                if cur_res in attention_resolutions:
                    self.down_blocks.append(Attention(channels))
            
            # 添加下采样层(除了最后一层)
            if level != len(channel_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
                )
                cur_res *= 2
        
        # 中间块，具有最大通道数
        self.mid_blocks = nn.ModuleList([
            Block(channels, channels, time_emb_dim=time_emb_dim),
            Attention(channels),
            Block(channels, channels, time_emb_dim=time_emb_dim)
        ])
        
        # 上采样部分
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        # 构建上采样层(从高层到低层)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # 每个分辨率有多个残差块
            for i in range(num_res_blocks + 1):
                # 连接来自下采样路径的特征
                in_ch = channels + channels if i == 0 and level != len(channel_mult) - 1 else channels
                out_ch = model_channels * mult
                
                # 添加残差块
                self.up_blocks.append(
                    Block(in_ch, out_ch, time_emb_dim=time_emb_dim, up=True)
                )
                channels = out_ch
                
                # 如果当前分辨率需要添加注意力层
                if cur_res in attention_resolutions:
                    self.up_blocks.append(Attention(channels))
            
            # 添加上采样层(除了最后一层)
            if level != 0:
                self.up_samples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                    )
                )
                cur_res //= 2
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 保存模块参考
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        
    def forward(self, x, time, condition=None):
        """
        前向传播
        
        参数:
            x: 输入张量
            time: 时间步
            condition: 条件输入(如果有)
            
        返回:
            预测的噪声
        """
        # 时间嵌入
        time_emb = self.time_embed(time)
        
        # 初始卷积
        h = self.input_conv(x)
        
        # 保存中间特征用于跳跃连接
        features = [h]
        
        # 下采样路径
        for i, layer in enumerate(self.down_blocks):
            h = layer(h, time_emb)
            
            # 保存每个残差块的输出
            if isinstance(layer, Block):
                features.append(h)
            
            # 应用下采样
            if i < len(self.down_samples) and len(self.down_blocks) - i - 1 >= len(self.down_samples):
                h = self.down_samples[i](h)
        
        # 中间块
        for layer in self.mid_blocks:
            if isinstance(layer, Block):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # 上采样路径
        block_counter = 0
        sample_counter = 0
        
        # 使用每个上采样块
        for i, layer in enumerate(self.up_blocks):
            if isinstance(layer, Block):
                # 当需要使用跳过连接时
                if block_counter == 0 and i > 0:
                    # 把下采样对应位置的特征连接起来
                    h = torch.cat([h, features.pop()], dim=1)
                
                h = layer(h, time_emb)
                block_counter += 1
                
                # 重置计数器
                if block_counter == 3:  # 2个残差块+1个可能的注意力块
                    block_counter = 0
                    
                    # 应用上采样
                    if sample_counter < len(self.up_samples):
                        h = self.up_samples[sample_counter](h)
                        sample_counter += 1
            else:
                h = layer(h)  # 处理注意力层
        
        # 输出层
        return self.output_conv(h) 