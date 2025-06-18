#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子动力学自由能计算的编码器-CNN模型

该模块实现了基于Vision Transformer架构的深度学习模型，专门用于处理λ窗口采样数据
并预测分子系统的自由能变化(ΔG)。

主要特性：
1. 自适应λ位置编码 - 可学习的lambda值位置编码机制
2. 多策略特征嵌入 - 支持3通道和4通道输入的不同嵌入策略  
3. 多损失函数组合 - 总ΔG损失、聚合窗口损失、平滑损失、特征损失
4. 动态数据标准化 - 基于训练集统计信息的实时标准化

模型架构基于Meta的Masked Autoencoder (MAE)，针对分子动力学数据进行了专门优化。

Author: 项目作者
Date: 2024
License: 基于Meta MAE项目许可证
"""

from functools import partial
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from utils.pos_embed import get_2d_sincos_pos_embed
from util.enc_model_dg_loss import dg_aggregation_loss_v2


class AdaptiveLambdaEncoding(nn.Module):
    """
    自适应λ位置编码模块
    
    该模块为不同的λ值提供可区分的位置编码，使用可学习的缩放因子C
    来适应λ值在[0,1]范围内的分布特性。
    
    基于正弦-余弦位置编码，但增加了可学习的缩放参数来优化编码效果。
    """
    
    def __init__(self, d_model: int, init_C: float = 100.0):
        """
        初始化自适应λ位置编码
        
        Args:
            d_model: 嵌入维度，必须是偶数
            init_C: 缩放因子的初始值，用于调节λ值编码的频率
        
        Raises:
            ValueError: 当d_model不是偶数时抛出异常
        """
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"嵌入维度d_model必须是偶数，当前为: {d_model}")
            
        self.d_model = d_model
        
        # 可学习的缩放因子C，用于调节λ值的编码频率
        self.C = nn.Parameter(torch.tensor(init_C, dtype=torch.float32))
        
        # 预计算频率衰减因子，用于生成不同频率的正弦余弦波
        # 使用标准Transformer位置编码的频率计算方式
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, lambda_val: torch.Tensor) -> torch.Tensor:
        """
        前向传播：为λ值生成位置编码
        
        Args:
            lambda_val: λ值张量，形状为 [N, H]，其中N是批次大小，H是窗口数(通常为100)
        
        Returns:
            pe: 位置编码张量，形状为 [N, H, d_model]
        """
        # 将λ值按可学习因子缩放，然后增加维度以便广播
        lambda_scaled = lambda_val.unsqueeze(-1) * self.C  # [N, H, 1]
        
        # 准备频率项用于广播计算
        div_term = self.div_term.to(lambda_val.device)  # [d_model/2]
        div_term_expanded = div_term.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model/2]
        
        # 初始化位置编码张量
        pe = torch.zeros(*lambda_val.shape, self.d_model, device=lambda_val.device, dtype=lambda_val.dtype)
        
        # 计算正弦和余弦编码
        # 偶数位置使用sin，奇数位置使用cos
        pe[..., 0::2] = torch.sin(lambda_scaled * div_term_expanded)  # [N, H, d_model/2]
        pe[..., 1::2] = torch.cos(lambda_scaled * div_term_expanded)  # [N, H, d_model/2]
        
        return pe


class BaseEmbeddingModule(nn.Module):
    """
    嵌入模块基类
    
    定义了所有嵌入策略必须实现的接口。不同的嵌入策略（如3通道、4通道）
    都继承自这个基类，确保接口的一致性。
    """
    
    def forward(self, 
                x: torch.Tensor,  # 窗口数据 (N, C, H, W) - 标准化后的μ、σ、error
                lambdas: torch.Tensor,  # λ值 (N, H)
                deltas: torch.Tensor,  # Δλ值 (N, H)
                masks: Dict[str, torch.Tensor],  # 掩码字典 (各种N×H张量)
                original_lengths: torch.Tensor  # 原始长度 (N, H)
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播接口
        
        Args:
            x: 输入数据张量
            lambdas: λ值张量
            deltas: Δλ值张量  
            masks: 掩码字典
            original_lengths: 原始数据长度张量
        
        Returns:
            tuple: (lambda_emb, feat_emb) - λ嵌入和特征嵌入
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现forward方法")


class EmbeddingStrategy3Chans(BaseEmbeddingModule):
    """
    3通道嵌入策略
    
    适用于输入通道数为3的情况，包含μ、σ²、error三个通道。
    使用Lambda特征投影和CNN编码器来生成嵌入表示。
    
    该策略将λ、Δλ和窗口掩码作为3个特征进行投影，
    同时使用CNN对数据特征进行编码。
    """
    
    def __init__(self, embed_dim: int, img_size: Tuple[int, int], in_chans: int = 3):
        """
        初始化3通道嵌入策略
        
        Args:
            embed_dim: 嵌入维度
            img_size: 输入图像尺寸 (H, W)，例如 (100, 50)
            in_chans: 输入通道数，必须为3
            
        Raises:
            ValueError: 当in_chans不等于3时抛出异常
        """
        super().__init__()
        
        if in_chans != 3:
            raise ValueError(f"EmbeddingStrategy3Chans要求in_chans=3，当前为: {in_chans}")
        
        self.embed_dim = embed_dim
        self.img_height, self.img_width = img_size  # 应该是 (100, 50)
        self.in_chans = in_chans  # 应该是 3
        
        # Lambda特征投影网络
        # 将lambda、delta、window mask作为3个特征进行投影
        self.lambda_proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # CNN编码器
        # 输入形状: [N*H, in_chans, 1, W] -> [N*100, 3, 1, 50]
        # 输出形状: [N*100, embed_dim, 1, 1] 经过AdaptiveAvgPool2d后
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_chans,  # 3个输入通道
                out_channels=self.embed_dim,
                kernel_size=(1, 3),  # 在宽度方向使用3x1卷积
                stride=(1, 1),
                padding=(0, 1)  # 在宽度方向padding以保持尺寸
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化到1x1
        )

    def forward(self, 
                x: torch.Tensor,  # [N, 3, 100, 50] (标准化后的μ、σ、error)
                lambdas: torch.Tensor,  # [N, 100]
                deltas: torch.Tensor,  # [N, 100]
                masks: Dict[str, torch.Tensor],  # {'window': [N, 100], ...}
                original_lengths: torch.Tensor  # [N, 100] (未使用但保留接口一致性)
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3通道嵌入的前向传播
        
        Args:
            x: 输入数据 [N, 3, 100, 50]
            lambdas: λ值 [N, 100]
            deltas: Δλ值 [N, 100]
            masks: 掩码字典
            original_lengths: 原始长度（此策略中未使用）
        
        Returns:
            tuple: (lambda_emb, feat_emb) 
                - lambda_emb: λ嵌入 [N, 100, embed_dim]
                - feat_emb: 特征嵌入 [N, 100, embed_dim]
        """
        N, C, H, W = x.shape  # C应该是3, H=100, W=50
        
        # 1. Lambda特征投影
        # 将lambda、delta和window mask堆叠作为特征进行投影
        lambda_feat = torch.stack([lambdas, deltas, masks['window']], dim=-1)  # [N, 100, 3]
        lambda_emb = self.lambda_proj(lambda_feat)  # [N, 100, embed_dim]
        
        # 2. CNN特征嵌入
        # 重塑数据用于CNN处理: [N, 3, 100, 50] -> [N*100, 3, 1, 50]
        # CNN期望的输入格式: [batch_size, channels, height, width]
        # 我们将每个窗口的数据作为单独的batch项目处理
        x_reshaped_for_cnn = x.permute(0, 2, 1, 3).reshape(N*H, C, 1, W)  # [N*100, 3, 1, 50]
        
        # 通过CNN编码器处理
        conv_out = self.cnn_encoder(x_reshaped_for_cnn)  # [N*100, embed_dim, 1, 1]
        
        # 移除大小为1的空间维度
        conv_out_aggregated = conv_out.squeeze(-1).squeeze(-1)  # [N*100, embed_dim]
        
        # 重塑回原始批次结构
        feat_emb = conv_out_aggregated.reshape(N, H, -1)  # [N, 100, embed_dim]
        
        return lambda_emb, feat_emb


class EmbeddingStrategy4Chans(BaseEmbeddingModule):
    """
    4通道嵌入策略
    
    适用于输入通道数为4的情况，包含μ、σ²、error和Δλ四个通道。
    使用自适应λ位置编码和多种特征编码方式。
    
    支持三种特征编码方式：
    1. 'cnn' - 使用CNN编码器
    2. 'linear' - 使用线性扁平化编码器  
    3. 'mlp' - 使用多层感知机编码器
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 img_size: Tuple[int, int], 
                 in_chans: int = 4,
                 embedding_strategy_type: str = 'cnn'):
        """
        初始化4通道嵌入策略
        
        Args:
            embed_dim: 嵌入维度
            img_size: 输入图像尺寸 (H, W)，例如 (100, 50)
            in_chans: 输入通道数，必须为4
            embedding_strategy_type: 嵌入策略类型 ('cnn', 'linear', 'mlp')
            
        Raises:
            ValueError: 当in_chans不等于4或embedding_strategy_type无效时抛出异常
        """
        super().__init__()
        
        if in_chans != 4:
            raise ValueError(f"EmbeddingStrategy4Chans要求in_chans=4，当前为: {in_chans}")
            
        valid_strategies = ['cnn', 'linear', 'mlp']
        if embedding_strategy_type not in valid_strategies:
            raise ValueError(f"嵌入策略类型必须是{valid_strategies}之一，当前为: {embedding_strategy_type}")
        
        self.embed_dim = embed_dim
        self.img_height, self.img_width = img_size  # 应该是 (100, 50)
        self.in_chans = in_chans  # 应该是 4
        self.embedding_strategy_type = embedding_strategy_type
        
        # 自适应λ位置编码
        self.adaptive_lambda_encoding = AdaptiveLambdaEncoding(embed_dim)
        
        # 根据策略类型创建相应的特征编码器
        if embedding_strategy_type == 'cnn':
            self._create_cnn_encoder()
        elif embedding_strategy_type == 'linear':
            self._create_linear_encoder()
        elif embedding_strategy_type == 'mlp':
            self._create_mlp_encoder()
    
    def _create_cnn_encoder(self):
        """创建CNN特征编码器"""
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_chans,  # 4个输入通道
                out_channels=self.embed_dim,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1)
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _create_linear_encoder(self):
        """创建线性扁平化编码器"""
        # 计算扁平化后的特征维度
        flattened_dim = self.in_chans * self.img_width
        self.feature_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),  # 扁平化 [C, W] -> [C*W]
            nn.Linear(flattened_dim, self.embed_dim),
            nn.ReLU()
        )
    
    def _create_mlp_encoder(self):
        """创建MLP特征编码器"""
        flattened_dim = self.in_chans * self.img_width
        self.feature_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU()
        )

    def forward(self, 
                x: torch.Tensor,  # [N, 4, 100, 50] (标准化的μ、σ、error + Δλ)
                lambdas: torch.Tensor,  # [N, 100]
                deltas: torch.Tensor,  # [N, 100] (未使用但保留接口一致性)
                masks: Dict[str, torch.Tensor],  # {'window': [N, 100], ...} (未使用但保留接口一致性)
                original_lengths: torch.Tensor  # [N, 100] (未使用但保留接口一致性)
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        4通道嵌入的前向传播
        
        Args:
            x: 输入数据 [N, 4, 100, 50]
            lambdas: λ值 [N, 100]
            deltas: Δλ值（此策略中未使用）
            masks: 掩码字典（此策略中未使用）
            original_lengths: 原始长度（此策略中未使用）
        
        Returns:
            tuple: (lambda_emb, feat_emb)
                - lambda_emb: λ嵌入 [N, 100, embed_dim]
                - feat_emb: 特征嵌入 [N, 100, embed_dim]
        """
        N, C, H, W = x.shape  # C应该是4, H=100, W=50
        
        # 1. 自适应λ位置编码
        lambda_emb = self.adaptive_lambda_encoding(lambdas)  # [N, 100, embed_dim]
        
        # 2. 特征编码
        if self.embedding_strategy_type == 'cnn':
            # CNN策略：类似3通道的处理方式
            x_reshaped = x.permute(0, 2, 1, 3).reshape(N*H, C, 1, W)  # [N*100, 4, 1, 50]
            feat_out = self.feature_encoder(x_reshaped)  # [N*100, embed_dim, 1, 1]
            feat_out = feat_out.squeeze(-1).squeeze(-1)  # [N*100, embed_dim]
            feat_emb = feat_out.reshape(N, H, -1)  # [N, 100, embed_dim]
        
        else:  # 'linear' 或 'mlp' 策略
            # 重塑为 [N*H, C, W] 然后通过编码器
            x_reshaped = x.permute(0, 2, 1, 3).reshape(N*H, C, W)  # [N*100, 4, 50]
            feat_out = self.feature_encoder(x_reshaped)  # [N*100, embed_dim]
            feat_emb = feat_out.reshape(N, H, -1)  # [N, 100, embed_dim]
        
        return lambda_emb, feat_emb


class MaskedAutoencoderViT(nn.Module):
    """
    分子动力学自由能预测的Masked Autoencoder Vision Transformer
    
    这是主要的模型类，整合了多种嵌入策略、Transformer编码器和投影头。
    模型支持3通道和4通道输入，并使用多种损失函数进行训练。
    
    主要组件：
    1. 嵌入模块 - 根据输入通道数选择合适的嵌入策略
    2. Transformer编码器 - 使用多头注意力机制处理序列数据
    3. 投影头 - 将编码器输出映射为3个特征(μ, σ, error)
    4. 损失计算 - 支持多种损失函数的组合
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (100, 50), 
                 patch_size: int = 16,
                 in_chans: int = 3, 
                 embed_dim: int = 1024, 
                 depth: int = 24, 
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512, 
                 decoder_depth: int = 8, 
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0, 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss: bool = False,
                 train_means: torch.Tensor = None, 
                 train_stds: torch.Tensor = None,
                 total_dg_loss_weight: float = 1.0,
                 agg_dg_loss_weight: float = 1.0, 
                 smoothness_loss_weight: float = 0.1,
                 feature_loss_weight: float = 1.0,
                 target_output_windows: int = 100):
        """
        初始化MAE模型
        
        Args:
            img_size: 输入图像尺寸 (H, W)
            patch_size: 补丁大小（未使用，保留兼容性）
            in_chans: 输入通道数 (3或4)
            embed_dim: 嵌入维度
            depth: Transformer编码器层数
            num_heads: 多头注意力头数
            decoder_embed_dim: 解码器嵌入维度（未使用，保留兼容性）
            decoder_depth: 解码器层数（未使用，保留兼容性）
            decoder_num_heads: 解码器注意力头数（未使用，保留兼容性）
            mlp_ratio: MLP隐藏层比例
            norm_layer: 归一化层类型
            norm_pix_loss: 是否使用归一化像素损失（未使用，保留兼容性）
            train_means: 训练数据均值 [3]
            train_stds: 训练数据标准差 [3]
            total_dg_loss_weight: 总ΔG损失权重
            agg_dg_loss_weight: 聚合ΔG损失权重
            smoothness_loss_weight: 平滑损失权重
            feature_loss_weight: 特征损失权重
            target_output_windows: 目标输出窗口数
        """
        super().__init__()
        
        # 基本参数
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.target_output_windows = target_output_windows
        
        # 损失权重
        self.total_dg_loss_weight = total_dg_loss_weight
        self.agg_dg_loss_weight = agg_dg_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight
        self.feature_loss_weight = feature_loss_weight
        
        # 注册训练数据统计信息
        if train_means is not None:
            self.register_buffer('train_means', train_means.clone())
        else:
            self.register_buffer('train_means', torch.zeros(3))
            
        if train_stds is not None:
            self.register_buffer('train_stds', train_stds.clone())
        else:
            self.register_buffer('train_stds', torch.ones(3))
        
        # 根据输入通道数选择嵌入策略
        if in_chans == 3:
            self.embedding_module = EmbeddingStrategy3Chans(embed_dim, img_size, in_chans)
        elif in_chans == 4:
            self.embedding_module = EmbeddingStrategy4Chans(embed_dim, img_size, in_chans, 'cnn')
        else:
            raise ValueError(f"不支持的输入通道数: {in_chans}。仅支持3或4通道。")
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # 投影头：将编码器输出映射为3个特征 (μ, σ, error)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3)  # 输出3个特征
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, target_output_windows, embed_dim))
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化模型权重"""
        # 初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.target_output_windows**0.5), 
            cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # 应用权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_encoder(self, 
                       processed_data_dict: Dict, 
                       original_window_data_list, 
                       original_window_lambdas_list, 
                       original_window_deltas_list,
                       original_window_lengths_list) -> torch.Tensor:
        """
        编码器前向传播
        
        Args:
            processed_data_dict: 处理后的数据字典
            original_window_data_list: 原始窗口数据列表（损失计算用）
            original_window_lambdas_list: 原始窗口λ值列表（损失计算用）
            original_window_deltas_list: 原始窗口Δλ值列表（损失计算用）
            original_window_lengths_list: 原始窗口长度列表（损失计算用）
        
        Returns:
            x: 编码后的特征张量 [N, target_output_windows, 3]
        """
        # 提取数据
        data = processed_data_dict['data']  # [N, C, 100, max_data]
        lambdas = processed_data_dict['lambdas']  # [N, 100]
        deltas = processed_data_dict['deltas']  # [N, 100]
        masks = processed_data_dict['masks']  # dict of masks
        original_lengths = processed_data_dict['original_lengths']  # [N, 100]
        
        # 数据标准化（仅对前3个通道）
        data_normalized = data.clone()
        for i in range(min(3, data.shape[1])):
            data_normalized[:, i] = (data[:, i] - self.train_means[i]) / self.train_stds[i]
        
        # 获取嵌入
        lambda_emb, feat_emb = self.embedding_module(
            data_normalized, lambdas, deltas, masks, original_lengths
        )
        
        # 组合嵌入
        x = lambda_emb + feat_emb  # [N, 100, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # 投影到3个特征
        x = self.projection_head(x)  # [N, 100, 3]
        
        return x
    
    def forward_loss(self, 
                    pred: torch.Tensor, 
                    processed_data_dict: Dict,
                    original_window_data_list, 
                    original_window_lambdas_list,
                    original_window_deltas_list, 
                    original_window_lengths_list,
                    original_window_dGs_list) -> Dict:
        """
        计算损失函数
        
        Args:
            pred: 模型预测 [N, 100, 3]
            processed_data_dict: 处理后的数据字典
            original_window_data_list: 原始窗口数据列表
            original_window_lambdas_list: 原始窗口λ值列表
            original_window_deltas_list: 原始窗口Δλ值列表
            original_window_lengths_list: 原始窗口长度列表
            original_window_dGs_list: 原始窗口ΔG值列表
        
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 反标准化预测值
        pred_denormalized = pred.clone()
        for i in range(3):
            pred_denormalized[:, :, i] = pred[:, :, i] * self.train_stds[i] + self.train_means[i]
        
        # 提取μ、σ、error预测
        pred_mu = pred_denormalized[:, :, 0]  # [N, 100]
        pred_sigma = pred_denormalized[:, :, 1]  # [N, 100]
        pred_error = pred_denormalized[:, :, 2]  # [N, 100]
        
        # 计算每窗口ΔG：dG = μ - σ²/2 + error
        pred_dg_per_window = pred_mu - pred_sigma**2 / 2 + pred_error  # [N, 100]
        
        # 物理常数
        kbt = 0.592  # kcal/mol
        
        # 初始化损失字典
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 总ΔG损失
        if self.total_dg_loss_weight > 0:
            window_mask = processed_data_dict['masks']['window']  # [N, 100]
            pred_total_dg = (pred_dg_per_window * window_mask).sum(dim=1) * kbt  # [N]
            
            # 计算目标总ΔG
            target_total_dgs = []
            for sample_dgs in original_window_dGs_list:
                target_total_dg = sum(sample_dgs) * kbt
                target_total_dgs.append(target_total_dg)
            target_total_dg = torch.tensor(target_total_dgs, device=pred.device, dtype=pred.dtype)
            
            total_dg_loss = F.mse_loss(pred_total_dg, target_total_dg)
            loss_dict['total_dg_loss'] = total_dg_loss
            total_loss += self.total_dg_loss_weight * total_dg_loss
        
        # 2. 聚合ΔG损失
        if self.agg_dg_loss_weight > 0:
            agg_dg_loss = dg_aggregation_loss_v2(
                pred_dg_per_window=pred_dg_per_window,
                window_mask=window_mask,
                original_window_lambdas_list=original_window_lambdas_list,
                original_window_deltas_list=original_window_deltas_list,
                original_window_dGs_list=original_window_dGs_list,
                reduction='mean'
            )
            loss_dict['agg_dg_loss'] = agg_dg_loss
            total_loss += self.agg_dg_loss_weight * agg_dg_loss
        
        # 3. 平滑损失
        if self.smoothness_loss_weight > 0:
            def compute_smoothness_loss(tensor, mask):
                """计算张量的平滑损失（基于二阶导数）"""
                if tensor.shape[1] < 3:
                    return torch.tensor(0.0, device=tensor.device)
                
                # 计算二阶导数
                second_derivative = tensor[:, 2:] - 2 * tensor[:, 1:-1] + tensor[:, :-2]
                
                # 应用掩码
                smoothness_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
                
                # 计算平滑损失
                if smoothness_mask.sum() > 0:
                    return (second_derivative**2 * smoothness_mask).sum() / smoothness_mask.sum()
                else:
                    return torch.tensor(0.0, device=tensor.device)
            
            mu_smoothness = compute_smoothness_loss(pred_mu, window_mask)
            sigma_smoothness = compute_smoothness_loss(pred_sigma, window_mask)
            
            smoothness_loss = mu_smoothness + sigma_smoothness
            loss_dict['smoothness_loss'] = smoothness_loss
            total_loss += self.smoothness_loss_weight * smoothness_loss
        
        # 4. 特征损失
        if self.feature_loss_weight > 0:
            # 准备目标特征
            target_mu_list = []
            target_sigma_list = []
            target_error_list = []
            
            for i, sample_data_list in enumerate(original_window_data_list):
                sample_lambdas = original_window_lambdas_list[i]
                
                # 创建目标特征张量
                target_mu_sample = torch.zeros(100, device=pred.device, dtype=pred.dtype)
                target_sigma_sample = torch.zeros(100, device=pred.device, dtype=pred.dtype)
                target_error_sample = torch.zeros(100, device=pred.device, dtype=pred.dtype)
                
                # 标准λ网格
                std_lambdas = np.arange(0, 1.0, 0.01)[:100]
                
                # 填充目标值
                for j, (data_tensor, lambda_val) in enumerate(zip(sample_data_list, sample_lambdas)):
                    target_idx = np.abs(std_lambdas - lambda_val).argmin()
                    if target_idx < 100:
                        target_mu_sample[target_idx] = data_tensor[0].mean()
                        target_sigma_sample[target_idx] = data_tensor[1].mean()
                        target_error_sample[target_idx] = data_tensor[2].mean()
                
                target_mu_list.append(target_mu_sample)
                target_sigma_list.append(target_sigma_sample)
                target_error_list.append(target_error_sample)
            
            target_mu = torch.stack(target_mu_list)  # [N, 100]
            target_sigma = torch.stack(target_sigma_list)  # [N, 100]
            target_error = torch.stack(target_error_list)  # [N, 100]
            
            # 计算特征损失（仅在有效窗口）
            mu_loss = F.mse_loss(pred_mu * window_mask, target_mu * window_mask)
            sigma_loss = F.mse_loss(pred_sigma * window_mask, target_sigma * window_mask)
            error_loss = F.mse_loss(pred_error * window_mask, target_error * window_mask)
            
            feature_loss = mu_loss + sigma_loss + error_loss
            loss_dict['feature_loss'] = feature_loss
            total_loss += self.feature_loss_weight * feature_loss
        
        loss_dict['loss'] = total_loss
        return loss_dict
    
    def forward(self, 
               processed_data_dict, 
               original_window_data_list, 
               original_window_lambdas_list,
               original_window_deltas_list, 
               original_window_lengths_list, 
               original_window_dGs_list):
        """
        模型前向传播
        
        Args:
            processed_data_dict: 处理后的数据字典
            original_window_data_list: 原始窗口数据列表
            original_window_lambdas_list: 原始窗口λ值列表
            original_window_deltas_list: 原始窗口Δλ值列表
            original_window_lengths_list: 原始窗口长度列表
            original_window_dGs_list: 原始窗口ΔG值列表
        
        Returns:
            loss_dict: 包含损失和预测的字典
        """
        # 编码器前向传播
        pred = self.forward_encoder(
            processed_data_dict, 
            original_window_data_list,
            original_window_lambdas_list, 
            original_window_deltas_list,
            original_window_lengths_list
        )
        
        # 计算损失
        loss_dict = self.forward_loss(
            pred, 
            processed_data_dict,
            original_window_data_list, 
            original_window_lambdas_list,
            original_window_deltas_list, 
            original_window_lengths_list,
            original_window_dGs_list
        )
        
        # 添加预测结果到返回字典
        loss_dict['predictions'] = pred
        
        return loss_dict


# 模型创建函数
def enc_cnn_chans3(**kwargs):
    """创建3通道CNN编码器模型"""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        in_chans=3, **kwargs
    )
    return model


def enc_cnn_chans4(**kwargs):
    """创建4通道CNN编码器模型"""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        in_chans=4, **kwargs
    )
    return model