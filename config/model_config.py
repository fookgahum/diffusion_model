"""
扩散模型配置文件
包含模型架构、扩散过程和采样相关的配置参数
"""

class DiffusionConfig:
    """扩散模型的基本配置类"""
    
    def __init__(self):
        # 模型基本参数
        self.model_dim = 256           # 模型基础维度
        self.time_emb_dim = 512        # 时间嵌入维度
        self.cond_emb_dim = 256        # 条件嵌入维度
        self.use_attention = True      # 是否使用注意力机制
        self.dropout = 0.1             # Dropout比率
        
        # U-Net参数
        self.unet_dim_mults = (1, 2, 4, 8)  # U-Net各层通道数倍率
        self.unet_resnet_blocks = 2    # 每层ResNet块数量
        
        # 扩散过程参数
        self.timesteps = 1000          # 扩散步数
        self.beta_schedule = 'linear'  # beta调度类型: 'linear', 'cosine', 'sigmoid'
        self.beta_start = 1e-4         # beta起始值
        self.beta_end = 0.02           # beta结束值
        
        # 采样参数
        self.sampling_steps = 100      # 采样步数(inference时可以少于训练步数)
        self.use_ddim = True           # 是否使用DDIM采样 
        self.clip_denoised = True      # 是否裁剪去噪后的值到[-1, 1]
        self.ddim_sampling_eta = 0.0   # DDIM采样的噪声系数(0=确定性)
    
    def __repr__(self):
        """打印配置信息"""
        config_str = "DiffusionConfig:\n"
        for attr, value in self.__dict__.items():
            config_str += f"  {attr}: {value}\n"
        return config_str


class TransformerConfig:
    """Transformer相关配置"""
    
    def __init__(self):
        self.n_heads = 8               # 注意力头数
        self.head_dim = 64             # 每个注意力头的维度
        self.context_dim = 512         # 交叉注意力上下文维度
        self.use_self_attention = True # 是否使用自注意力
        self.use_cross_attention = True # 是否使用交叉注意力
        self.attention_dropout = 0.1   # 注意力的dropout比率


class TrainingConfig:
    """训练相关配置"""
    
    def __init__(self):
        self.batch_size = 32           # 批次大小
        self.learning_rate = 2e-5      # 学习率
        self.weight_decay = 1e-4       # 权重衰减
        self.ema_decay = 0.9999        # EMA衰减率
        self.gradient_clip = 1.0       # 梯度裁剪
        self.lr_warmup_steps = 500     # 学习率预热步数


# 默认配置
DEFAULT_CONFIG = DiffusionConfig() 