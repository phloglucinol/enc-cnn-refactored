#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

该模块定义了整个项目的配置结构，包括：
1. 数据配置 - 数据路径、预处理参数等
2. 模型配置 - 模型架构、超参数等  
3. 训练配置 - 训练策略、优化器设置等
4. 输出配置 - 日志、检查点保存等

通过统一的配置管理，提高代码的可维护性和可扩展性。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import torch


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    data_path: str = '/nfs/export4_25T/ynlu/data/enc_cnn_dU_info_dataset/8-1-1/s0'
    
    # 数据预处理配置
    subset_size: int = 14  # 子集大小
    num_random_subsets_per_system: int = 20  # 每个系统生成的随机子集数量
    per_lambda_max_points: int = 10  # 每个lambda窗口最大数据点数
    processor_max_data: int = 50  # 数据处理器最大数据点数
    processor_include_delta: bool = True  # 是否包含delta_lambda作为第4通道
    
    # 数据加载配置
    num_workers: int = 8  # 数据加载器工作进程数
    pin_memory: bool = True  # 是否将数据固定在内存
    

@dataclass  
class ModelConfig:
    """模型相关配置"""
    # 模型基本参数
    model_name: str = 'enc_cnn_chans3'  # 模型名称
    input_size: Tuple[int, int] = (50, 100)  # 输入尺寸 (高度, 宽度)
    in_chans: int = 3  # 输入通道数
    
    # 模型架构参数
    embed_dim: int = 768  # 嵌入维度
    depth: int = 12  # Transformer层数
    num_heads: int = 12  # 多头注意力头数
    
    # 损失函数权重
    total_dg_loss_weight: float = 1.0  # 总ΔG损失权重
    agg_dg_loss_weight: float = 1.0  # 聚合窗口ΔG损失权重
    smoothness_loss_weight: float = 0.1  # 平滑损失权重
    feature_loss_weight: float = 1.0  # 特征损失权重
    
    # 其他模型参数
    norm_pix_loss: bool = False  # 是否使用归一化像素损失


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基本训练参数
    batch_size: int = 4  # 批次大小
    epochs: int = 100  # 训练轮数
    accum_iter: int = 1  # 梯度累积迭代次数
    
    # 优化器配置
    lr: Optional[float] = None  # 学习率（如果为None则自动计算）
    blr: float = 1e-3  # 基础学习率
    min_lr: float = 0.0  # 最小学习率
    weight_decay: float = 0.05  # 权重衰减
    warmup_epochs: int = 40  # 学习率预热轮数
    
    # 训练策略
    start_epoch: int = 0  # 开始轮数
    resume: str = ''  # 恢复训练的检查点路径
    
    # 设备配置
    device: str = 'cuda'  # 训练设备
    seed: int = 0  # 随机种子
    
    # 分布式训练配置
    world_size: int = 1  # 分布式进程数
    local_rank: int = -1  # 本地rank
    dist_on_itp: bool = False  # 是否在ITP上分布式训练
    dist_url: str = 'env://'  # 分布式训练URL


@dataclass
class OutputConfig:
    """输出相关配置"""
    # 输出路径
    output_dir: str = './output_dir'  # 模型保存路径
    log_dir: str = './output_dir'  # 日志保存路径
    
    # 保存策略
    save_checkpoint_freq: int = 10  # 检查点保存频率（轮数）
    keep_best_model: bool = True  # 是否保存最佳模型
    
    # 日志配置
    log_freq: int = 10  # 日志打印频率（批次）
    tensorboard_log: bool = True  # 是否使用TensorBoard日志


@dataclass
class Config:
    """完整的项目配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """配置初始化后的处理"""
        # 确保输出目录存在
        os.makedirs(self.output.output_dir, exist_ok=True)
        os.makedirs(self.output.log_dir, exist_ok=True)
        
        # 自动计算学习率
        if self.training.lr is None:
            # 根据有效批次大小计算学习率
            eff_batch_size = (self.training.batch_size * 
                            self.training.accum_iter * 
                            self.training.world_size)
            self.training.lr = self.training.blr * eff_batch_size / 256
    
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置对象"""
        config = cls()
        
        # 更新数据配置
        config.data.data_path = args.data_path
        config.data.subset_size = args.subset_size
        config.data.num_random_subsets_per_system = args.num_random_subsets_per_system
        config.data.per_lambda_max_points = args.per_lambda_max_points
        config.data.processor_include_delta = args.processor_include_delta
        config.data.num_workers = args.num_workers
        config.data.pin_memory = args.pin_mem
        
        # 更新模型配置
        config.model.model_name = args.model
        config.model.input_size = args.input_size
        config.model.in_chans = args.in_chans
        config.model.total_dg_loss_weight = args.total_dg_loss_weight
        config.model.agg_dg_loss_weight = args.agg_dg_loss_weight
        config.model.smoothness_loss_weight = args.smoothness_loss_weight
        config.model.feature_loss_weight = args.feature_loss_weight
        config.model.norm_pix_loss = args.norm_pix_loss
        
        # 更新训练配置
        config.training.batch_size = args.batch_size
        config.training.epochs = args.epochs
        config.training.accum_iter = args.accum_iter
        config.training.lr = args.lr
        config.training.blr = args.blr
        config.training.min_lr = args.min_lr
        config.training.weight_decay = args.weight_decay
        config.training.warmup_epochs = args.warmup_epochs
        config.training.start_epoch = args.start_epoch
        config.training.resume = args.resume
        config.training.device = args.device
        config.training.seed = args.seed
        config.training.world_size = args.world_size
        config.training.local_rank = args.local_rank
        config.training.dist_on_itp = args.dist_on_itp
        config.training.dist_url = args.dist_url
        
        # 更新输出配置
        config.output.output_dir = args.output_dir
        config.output.log_dir = args.log_dir
        
        return config
    
    def validate(self):
        """验证配置的有效性"""
        # 验证数据配置
        if not os.path.exists(self.data.data_path):
            raise ValueError(f"数据路径不存在: {self.data.data_path}")
        
        # 验证模型配置
        if self.model.in_chans not in [3, 4]:
            raise ValueError(f"输入通道数必须为3或4，当前为: {self.model.in_chans}")
        
        # 验证训练配置
        if self.training.batch_size <= 0:
            raise ValueError(f"批次大小必须大于0，当前为: {self.training.batch_size}")
        
        if self.training.epochs <= 0:
            raise ValueError(f"训练轮数必须大于0，当前为: {self.training.epochs}")
        
        # 验证设备可用性
        if self.training.device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，自动切换到CPU")
            self.training.device = 'cpu'
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print("项目配置信息")
        print("=" * 50)
        
        print(f"\n[数据配置]")
        print(f"数据路径: {self.data.data_path}")
        print(f"子集大小: {self.data.subset_size}")
        print(f"每系统随机子集数: {self.data.num_random_subsets_per_system}")
        print(f"每lambda窗口最大点数: {self.data.per_lambda_max_points}")
        print(f"是否包含delta通道: {self.data.processor_include_delta}")
        
        print(f"\n[模型配置]")
        print(f"模型名称: {self.model.model_name}")
        print(f"输入尺寸: {self.model.input_size}")
        print(f"输入通道数: {self.model.in_chans}")
        print(f"损失权重 - 总ΔG: {self.model.total_dg_loss_weight}")
        print(f"损失权重 - 聚合ΔG: {self.model.agg_dg_loss_weight}")
        print(f"损失权重 - 平滑: {self.model.smoothness_loss_weight}")
        print(f"损失权重 - 特征: {self.model.feature_loss_weight}")
        
        print(f"\n[训练配置]")
        print(f"批次大小: {self.training.batch_size}")
        print(f"训练轮数: {self.training.epochs}")
        print(f"学习率: {self.training.lr:.6f}")
        print(f"权重衰减: {self.training.weight_decay}")
        print(f"预热轮数: {self.training.warmup_epochs}")
        print(f"训练设备: {self.training.device}")
        
        print(f"\n[输出配置]")
        print(f"输出目录: {self.output.output_dir}")
        print(f"日志目录: {self.output.log_dir}")
        
        print("=" * 50)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config.from_args(args)
    config.validate()
    return config