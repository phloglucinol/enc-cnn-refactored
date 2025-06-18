"""分子动力学自由能计算的编码器-CNN模型实现

本模块实现了基于Vision Transformer架构的自由能预测模型，专门用于处理λ窗口数据。
主要特性包括：
1. 自适应λ位置编码
2. 多策略特征嵌入（3通道/4通道）
3. 多损失函数组合（总dG损失、聚合窗口损失、平滑损失、特征损失）
4. 动态数据标准化

Author: Your Name
Date: 2024
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import List, Dict, Sequence, Tuple

from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed_test import get_2d_sincos_pos_embed
from util.enc_model_dg_loss import dg_aggregation_loss_v2
from util.test_lambda_emb_dataset import LambdaDataProcessor


# # data预处理模块 在dataset里实现
# class LambdaDataProcessor:
#     def __init__(self, min_delta=0.01, max_windows=100, max_data=100, include_delta_in_data=False):
#         """
#         :param min_delta: 补全窗口的Δλ值（固定0.01）
#         :param max_windows: 目标标准网格的总窗口数（例如 100）
#         :param max_data: 目标标准网格的每窗口最大数据点数（例如 40 或 100）
#         :param include_delta_in_data: 是否包含原始的Δλ数据
#         """
#         self.min_delta = min_delta
#         self.max_windows = max_windows
#         self.max_data = max_data
#         self.include_delta_in_data = include_delta_in_data

#         # 若需要λ=1.00，应调整max_windows=101
#         # 标准 λ 网格，max_windows 个点：[0.00, 0.01, 0.02, ..., 0.99]  # 共 100 个点
#         self.std_lambdas = np.round(np.arange(0, 1.0, min_delta), 4)[:max_windows]  # λ从0.00到0.99

# # --------------------------------------------------------
#     def process(self, original_data: Sequence[Sequence[torch.Tensor]],
#                 original_lambdas: Sequence[Sequence[float]],
#                 original_deltas: Sequence[Sequence[float]],
#                 original_data_lengths: Sequence[Sequence[int]]) -> Dict:
#         """
#         返回修改后的字典结构：
#         {
#             'data': torch.Tensor [N, 3, max_windows, max_data],  # 仅包含原始特征（μ、σ²、误差）
#             'lambdas': torch.Tensor [N, max_windows],            # λ值，空缺处填充标准λ
#             'deltas': torch.Tensor [N, max_windows],             # Δλ值，空缺处填充-1
#             'masks': {
#                 'window': torch.Tensor [N, max_windows],        # 有效窗口标记
#                 'delta': torch.Tensor [N, max_windows]          # 真实Δλ标记（非补全）
#             },
#             'original_lengths': torch.Tensor [N, max_windows],     # 原始数据长度
#             'raw_timeseries_lengths': torch.Tensor [N, max_windows] # 原始时间序列长度
#         }
#         """
#         batch_size = len(original_data)
#         if batch_size == 0:
#             return {
#                 'data': torch.empty(0, 3, self.max_windows, self.max_data),
#                 'lambdas': torch.empty(0, self.max_windows),
#                 'deltas': torch.empty(0, self.max_windows),
#                 'masks': {
#                     'window': torch.empty(0, self.max_windows),
#                     'delta': torch.empty(0, self.max_windows)
#                 },
#                 'original_lengths': torch.empty(0, self.max_windows)
#             }

#         device = 'cpu'
#         if batch_size > 0 and len(original_data[0]) > 0 and isinstance(original_data[0][0], torch.Tensor):
#             device = original_data[0][0].device

#         # 初始化输出张量（仅3通道）
#         aligned_data = torch.zeros(batch_size, 3, self.max_windows, self.max_data, device=device)
#         aligned_lambdas = torch.zeros(batch_size, self.max_windows, device=device)
#         aligned_deltas = torch.full((batch_size, self.max_windows), -1.0, device=device)  # 空缺处填充-1

#         masks = {
#             'window': torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device),
#             'delta': torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device)
#         }

#         std_lambdas_tensor = torch.from_numpy(self.std_lambdas).to(dtype=torch.float32, device=device)

#         for i in range(batch_size):
#             lambdas = original_lambdas[i]
#             deltas = original_deltas[i]
#             sample_raw_data = original_data[i]
#             sample_data_lengths = original_data_lengths[i]

#             # 输入验证（与之前相同）
#             num_windows_actual = len(lambdas)
#             assert num_windows_actual == len(deltas), f"样本 {i}: λ 数量 ({num_windows_actual}) 与 Δλ 数量 ({len(deltas)}) 不匹配"
#             assert num_windows_actual == len(sample_raw_data), f"样本 {i}: λ/Δλ 数量 ({num_windows_actual}) 与 原始数据张量列表长度 ({len(sample_raw_data)}) 不匹配"
#             assert num_windows_actual == len(sample_data_lengths), f"样本 {i}: λ/Δλ 数量 ({num_windows_actual}) 与 数据点数列表长度 ({len(sample_data_lengths)}) 不匹配"

#             for orig_idx in range(num_windows_actual):
#                 lbda = lambdas[orig_idx]
#                 delta = deltas[orig_idx]
#                 current_window_data = sample_raw_data[orig_idx].to(device)
#                 current_data_length = sample_data_lengths[orig_idx]

#                 # 数据验证（与之前相同）
#                 assert current_window_data.shape[0] == 3, f"样本 {i}, 原始窗口 {orig_idx}: 数据张量通道数应为 3, 实际为 {current_window_data.shape[0]}"
#                 assert current_window_data.shape[1] == current_data_length, f"样本 {i}, 原始窗口 {orig_idx}: 数据张量点数 ({current_window_data.shape[1]}) 与报告长度 ({current_data_length}) 不匹配"

#                 num_points_to_copy = min(current_data_length, self.max_data)
#                 if current_data_length > self.max_data:
#                     print(f"警告: 样本 {i}, 原始窗口 {orig_idx} (λ={lbda:.2f}) 的数据点数 ({current_data_length}) 超过 max_data ({self.max_data}). 数据将被截断.")
#                     current_window_data = current_window_data[:, :self.max_data]

#                 target_idx = np.abs(self.std_lambdas - lbda).argmin()
#                 if target_idx < self.max_windows:
#                     # 仅复制原始特征（前3通道）
#                     aligned_data[i, :3, target_idx, :num_points_to_copy] = current_window_data

#                     # 记录λ和Δλ（单独存储）
#                     aligned_lambdas[i, target_idx] = lbda
#                     aligned_deltas[i, target_idx] = delta  # 有效Δλ覆盖-1

#                     # 设置掩码
#                     masks['window'][i, target_idx] = 1
#                     masks['delta'][i, target_idx] = 1

#             # 用标准λ填充空缺位置（对齐后的λ始终完整）
#             aligned_lambdas[i] = std_lambdas_tensor

#         result = {
#             'data': aligned_data,      # [N, 3, 100, 50]
#             'lambdas': aligned_lambdas,  # [N, 100]
#             'deltas': aligned_deltas,    # [N, 100]（有效处为原始Δλ，空缺处为-1）
#             'masks': masks
#         }
        
#         # 添加原始长度信息 - 这是关键的新增部分
#         original_lengths_tensor = torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device)
        
#         for i in range(batch_size):
#             sample_data_lengths = original_data_lengths[i]
#             lambdas = original_lambdas[i]
            
#             # 记录每个窗口的原始数据点数量
#             for orig_idx in range(len(lambdas)):
#                 lbda = lambdas[orig_idx]
#                 orig_length = sample_data_lengths[orig_idx]
                
#                 # 找到标准网格中最接近的λ索引
#                 target_idx = np.abs(self.std_lambdas - lbda).argmin()
#                 if target_idx < self.max_windows:
#                     original_lengths_tensor[i, target_idx] = orig_length
        
#         # 将原始长度添加到返回结果
#         result['original_lengths'] = original_lengths_tensor

#         return result

# ---------------------------------------------------------

# 1. λ Positional Encoding 类
class AdaptiveLambdaEncoding(nn.Module):
    def __init__(self, d_model, init_C=100.0):
        super().__init__()
        self.d_model = d_model
        # 可学习的缩放因子C
        self.C = nn.Parameter(torch.tensor(init_C))
        # 预计算频率衰减因子 (这里使用的是标准的Sin-Cos PE公式的div_term)
        # d_model 必须是偶数
        if d_model % 2 != 0:
             raise ValueError(f"d_model must be an even number, but got {d_model}")
        self.register_buffer(
            "div_term",
            torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        )

    def forward(self, lambda_val: torch.Tensor):
        """
        Args:
            lambda_val: λ 值张量, 形状 [N, H] (H=100)
        Returns:
            pe: Positional Encoding 张量, 形状 [N, H, d_model]
        """
        # 对 lambda_val 进行缩放
        # Unsqueeze lambda_val to [N, H, 1] to allow broadcasting with div_term [d_model/2] -> [1, 1, d_model/2]
        lambda_scaled = lambda_val.unsqueeze(-1) * self.C # [N, H, 1] * scalar -> [N, H, 1]

        # Apply sine to even indices, cosine to odd indices
        div_term = self.div_term.to(lambda_val.device) # [d_model/2]
        # Expand div_term for broadcasting: [1, 1, d_model/2]
        div_term_expanded = div_term.unsqueeze(0).unsqueeze(0)

        # Resulting PE shape [N, H, d_model]
        pe = torch.zeros(*lambda_val.shape, self.d_model, device=lambda_val.device)

        # Apply sin and cos
        # Use lambda_scaled [N, H, 1] and div_term_expanded [1, 1, d_model/2]
        pe[..., 0::2] = torch.sin(lambda_scaled * div_term_expanded) # Broadcasting handles [N, H, 1] * [1, 1, d_model/2] -> [N, H, d_model/2]
        pe[..., 1::2] = torch.cos(lambda_scaled * div_term_expanded) # Broadcasting handles [N, H, 1] * [1, 1, d_model/2] -> [N, H, d_model/2]

        return pe


# ---------------------------------------------------------
# 2. 嵌入模块基类
class BaseEmbeddingModule(nn.Module):
    def forward(self, x: torch.Tensor, # Window data (N, C, H, W) - Assumed normalized μ, σ, error (first 3 channels)
                        lambdas: torch.Tensor, # Lambda values (N, H)
                        deltas: torch.Tensor, # Delta values (N, H)
                        masks: Dict[str, torch.Tensor], # Masks (dict of N, H tensors)
                        original_lengths: torch.Tensor # Original lengths (N, H)
                       ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns lambda_emb (N, H, D), feat_emb (N, H, D)
        """
        Base class for embedding modules. Subclasses must implement this forward method.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


# ---------------------------------------------------------
# 3. EmbeddingStrategy3Chans 模块
class EmbeddingStrategy3Chans(BaseEmbeddingModule):
    def __init__(self, embed_dim: int, img_size: Tuple[int, int], in_chans: int = 3):
        """
        Embedding strategy for in_chans=3, using Lambda Projection and CNN Encoder.
        img_size is (H, W) for the processed data, e.g., (100, 50).
        """
        super().__init__()
        # Validate in_chans
        if in_chans != 3:
            raise ValueError(f"EmbeddingStrategy3Chans requires in_chans=3, but got {in_chans}")

        self.embed_dim = embed_dim
        self.img_height, self.img_width = img_size # Should be (100, 50)
        self.in_chans = in_chans # Should be 3

        # Lambda Feature Projection (same as original)
        # Takes lambda, delta, window mask as 3 features
        self.lambda_proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # CNN Encoder (same as original, input channels = 3)
        # Input shape after permute/reshape: [N*H, in_chans, 1, W] -> [N*100, 3, 1, 50]
        # Output shape after AdaptiveAvgPool2d((1,1)): [N*100, embed_dim, 1, 1]
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chans, # Use self.in_chans (which is 3)
                     out_channels=self.embed_dim,
                     kernel_size=(1, 3),
                     stride=(1, 1),
                     padding=(0, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Note: Weight initialization for these layers will be handled by MaskedAutoencoderViT._init_weights via self.apply()


    def forward(self, x: torch.Tensor, # [N, 3, 100, 50] (Normalized μ, σ, error)
                        lambdas: torch.Tensor, # [N, 100]
                        deltas: torch.Tensor, # [N, 100]
                        masks: Dict[str, torch.Tensor], # {'window': [N, 100], ...}
                        original_lengths: torch.Tensor # [N, 100]
                       ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns lambda_emb [N, 100, D], feat_emb [N, 100, D]
        """
        Forward pass for 3-channel embedding.
        """
        N, C, H, W = x.shape # C should be 3, H=100, W=50

        # 1. Lambda Feature Projection
        # Use lambda, delta, and window mask as features for projection
        lambda_feat = torch.stack([lambdas, deltas, masks['window']], dim=-1)  # [N, 100, 3]
        lambda_emb = self.lambda_proj(lambda_feat)  # [N, 100, embed_dim]

        # 2. CNN Feature Embedding
        # Reshape for CNN: [N, 3, 100, 50] -> [N*100, 3, 1, 50]
        # Permute H (dim 2) and C (dim 1) to get [N, 100, 3, 50], then reshape to [N*100, 3, 50]
        # The CNN expects [N_batches, C_in, H_in, W_in]. With kernel (1, 3) and AdaptiveAvgPool2d((1,1)),
        # the H_in dimension is expected to be 1 for each "row" of data we feed in.
        # So, we process each window (C, W) separately as a batch item for the CNN.
        # Original: [N, C, H, W] -> reshape [N*H, C, 1, W]
        x_reshaped_for_cnn = x.permute(0, 2, 1, 3).reshape(N*H, C, 1, W) # [N*100, 3, 1, 50]

        conv_out = self.cnn_encoder(x_reshaped_for_cnn) # [N*100, embed_dim, 1, 1]

        # Remove spatial dimensions of size 1
        conv_out_aggregated = conv_out.squeeze(-1).squeeze(-1) # [N*100, embed_dim]

        # Reshape back to [N, 100, embed_dim]
        feat_emb = conv_out_aggregated.reshape(N, H, -1) # [N, 100, embed_dim]

        return lambda_emb, feat_emb


# ---------------------------------------------------------
# 4. EmbeddingStrategy4Chans 模块
class EmbeddingStrategy4Chans(BaseEmbeddingModule):
    def __init__(self, embed_dim: int, img_size: Tuple[int, int], in_chans: int = 4,
                 embedding_strategy_type: str = 'cnn'):
        """
        Embedding strategy for in_chans=4, using Lambda Positional Encoding and different
        feature embedding strategies ('cnn', 'flatten_linear', 'mlp').
        img_size is (H, W) for the processed data, e.g., (100, 50).
        """
        super().__init__()
        # Validate in_chans
        if in_chans != 4:
            raise ValueError(f"EmbeddingStrategy4Chans requires in_chans=4, but got {in_chans}")

        self.embed_dim = embed_dim
        self.img_height, self.img_width = img_size # Should be (100, 50)
        self.in_chans = in_chans # Should be 4
        self.embedding_strategy_type = embedding_strategy_type

        # 1. Lambda Positional Encoding (for in_chans=4)
        # Uses AdaptiveLambdaEncoding, takes lambda values [N, 100] and outputs [N, 100, embed_dim]
        self.lambda_pos_encoder = AdaptiveLambdaEncoding(d_model=embed_dim)

        # 2. Window Feature Embedding - Select strategy based on embedding_strategy_type
        if self.embedding_strategy_type == 'cnn':
            # CNN Encoder (same structure as 3-chans, but input channels = 4)
            # Input shape after permute/reshape: [N*H, in_chans, 1, W] -> [N*100, 4, 1, 50]
            # Output shape after AdaptiveAvgPool2d((1,1)): [N*100, embed_dim, 1, 1]
            self.feat_encoder = nn.Sequential(
                nn.Conv2d(in_channels=self.in_chans, # Use self.in_chans (which is 4)
                         out_channels=self.embed_dim,
                         kernel_size=(1, 3),
                         stride=(1, 1),
                         padding=(0, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif self.embedding_strategy_type == 'flatten_linear':
            # Flatten the last two dimensions (channels and max_data) and apply a Linear layer
            # Input to Flatten: [N*H, C, W] -> [N*H, C*W]
            # Input to Linear: [N*H, 4 * 50] = [N*H, 200]
            # Output of Linear: [N*H, embed_dim]
            flattened_size = self.in_chans * self.img_width # 4 * 50 = 200
            self.feat_encoder = nn.Sequential(
                nn.Flatten(start_dim=-2), # Flatten from channel dim (-2) to max_data dim (-1)
                nn.Linear(flattened_size, embed_dim)
            )
        elif self.embedding_strategy_type == 'mlp':
            # MLP Per Window: Flatten C and W for each window, then use an MLP
            # Input to MLP: [N*H, C*W] -> [N*100, 200]
            # Output of MLP: [N*100, embed_dim]
            # This is very similar to flatten_linear, but allows for a deeper MLP
            flattened_size = self.in_chans * self.img_width # 4 * 50 = 200
            self.feat_encoder = nn.Sequential(
                nn.Flatten(start_dim=-2), # Flatten from channel dim (-2) to max_data dim (-1)
                nn.Linear(flattened_size, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
        else:
            raise ValueError(f"Unsupported embedding_strategy_type for in_chans=4: {embedding_strategy_type}")

        # Note: Weight initialization for these layers will be handled by MaskedAutoencoderViT._init_weights via self.apply()

    def forward(self, x: torch.Tensor, # [N, 4, 100, 50] (Normalized μ, σ, error + original delta)
                        lambdas: torch.Tensor, # [N, 100]
                        deltas: torch.Tensor, # [N, 100]
                        masks: Dict[str, torch.Tensor], # {'window': [N, 100], ...}
                        original_lengths: torch.Tensor # [N, 100]
                       ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns lambda_emb [N, 100, D], feat_emb [N, 100, D]
        """
        Forward pass for 4-channel embedding with different feature strategies.
        """
        N, C, H, W = x.shape # C should be 4, H=100, W=50

        # 1. Lambda Positional Encoding
        # Only pass the lambda values to the positional encoder
        lambda_emb = self.lambda_pos_encoder(lambdas)  # [N, 100, embed_dim]

        # 2. Window Feature Embedding
        # Process x through the selected feature encoder
        if self.embedding_strategy_type == 'cnn':
            # Reshape for CNN: [N, 4, 100, 50] -> [N*100, 4, 1, 50]
            x_reshaped_for_feat_encoder = x.permute(0, 2, 1, 3).reshape(N*H, C, 1, W) # [N*100, 4, 1, 50]
            feat_emb = self.feat_encoder(x_reshaped_for_feat_encoder) # [N*100, embed_dim, 1, 1]
            # Remove spatial dimensions of size 1
            feat_emb = feat_emb.squeeze(-1).squeeze(-1) # [N*100, embed_dim]

        elif self.embedding_strategy_type in ['flatten_linear', 'mlp']:
            # Reshape for Flatten/MLP: [N, 4, 100, 50] -> [N*100, 4, 50]
            x_reshaped_for_feat_encoder = x.permute(0, 2, 1, 3).reshape(N*H, C, W) # [N*100, 4, 50]
            feat_emb = self.feat_encoder(x_reshaped_for_feat_encoder) # [N*100, embed_dim]
            # Output is already [N*H, embed_dim], no need to squeeze spatial dims

        else:
             # This should be caught in __init__, but good to have a fallback
             raise ValueError(f"Unsupported embedding_strategy_type: {self.embedding_strategy_type}")

        # Reshape feature embedding back to [N, 100, embed_dim]
        feat_emb = feat_emb.reshape(N, H, -1) # [N, 100, embed_dim]


        return lambda_emb, feat_emb


# ---------------------------------------------------------
# MaskedAutoencoderViT class
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=(100, 50), in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 target_output_windows=100,
                 total_dg_loss_weight=1.0,
                 agg_dg_loss_weight=1.0,
                 train_means: torch.Tensor = None, # Add train_means
                 train_stds: torch.Tensor = None, # Add train_stds
                 # Add parameter for embedding strategy type for in_chans=4
                 embedding_strategy_type: str = 'cnn',
                 # 新增平滑损失权重参数
                 smoothness_loss_weight: float = 0.0,
                 # Add parameter for feature loss weight
                 feature_loss_weight: float = 0.0
                 ):
        super().__init__()

        # self.patch_size = patch_size
        self.img_size = img_size # (H, W) e.g., (100, 50)
        self.in_chans = in_chans # 3 or 4
        self.embed_dim = embed_dim
        self.target_output_windows = target_output_windows # Should be 100

        # Loss weights
        self.total_dg_loss_weight = total_dg_loss_weight
        self.agg_dg_loss_weight = agg_dg_loss_weight
        # Store smoothness loss weight
        self.smoothness_loss_weight = smoothness_loss_weight
        # Add parameter for feature loss weight
        self.feature_loss_weight = feature_loss_weight

        # Store training statistics
        # Ensure they are buffers and not parameters if they shouldn't be updated by gradients
        # They should be on the same device as the model parameters
        # Store training statistics as buffers
        if train_means is not None and train_stds is not None:
            assert train_means.shape == (3,) and train_stds.shape == (3,), "train_means and train_stds must have shape (3,)"
            # Ensure they are on CPU initially or handle device transfer
            # Use register_buffer to store these as part of the model state, but not trainable parameters
            self.register_buffer('train_means', train_means.clone().detach().cpu())
            self.register_buffer('train_stds', train_stds.clone().detach().cpu())
            print(f"Model initialized with train_means: {self.train_means.cpu().numpy()}")
            print(f"Model initialized with train_stds:  {self.train_stds.cpu().numpy()}")
        else:
             # Initialize with defaults if not provided (e.g., for non-training usage or errors)
             self.register_buffer('train_means', torch.zeros(3))
             self.register_buffer('train_stds', torch.ones(3))
             print("Warning: Model initialized without train_means/stds. Using defaults (0, 1).")


        # # --- Lambda Feature Projection ---
        # self.lambda_proj = nn.Sequential(
        #     nn.Linear(3, embed_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim // 2, embed_dim)
        # )
        
        # --- Embedding Module Selection ---
        # Instantiate the appropriate embedding module based on in_chans
        if in_chans == 3:
            self.embedding_module = EmbeddingStrategy3Chans(
                embed_dim=embed_dim,
                img_size=img_size,
                in_chans=in_chans
            )
        elif in_chans == 4:
            # Instantiate EmbeddingStrategy4Chans, passing the selected strategy type
            self.embedding_module = EmbeddingStrategy4Chans(
                embed_dim=embed_dim,
                img_size=img_size,
                in_chans=in_chans, # Pass 4
                embedding_strategy_type=embedding_strategy_type # Pass the strategy type
            )
        else:
            raise ValueError(f"Unsupported in_chans: {in_chans}")


        # # --- CNN Encoder ---
        # self.cnn_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_chans,
        #              out_channels=embed_dim,
        #              kernel_size=(1, 3),
        #              stride=(1, 1),
        #              padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
             
        # Note: Removed original lambda_proj and cnn_encoder as they are now inside embedding_module

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --- Projection Head ---
        # Output dimension is 3 (μ, σ, error)
        # The projection head maps from Transformer output (embed_dim)
        # to the features needed for dG calculation (3: μ, σ, error)
        # Depending on the chosen strategy (A or B), this head might predict
        # normalized or unnormalized features. Here we design it to output 3 features.
        # According to Option A, this head predicts NORMALIZED features.
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3)  # 直接输出3个特征 (预测归一化后的 μ, σ, error)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization---no need for this model now---
        # initialize (and freeze) pos_embed by sin-cos embedding
        # Note: Assuming a different positional embedding or none is used now,
        # as patch_size is not present. If positional embedding is needed for the
        # 100 windows, it should be handled differently.
        # The following lines are likely remnants from a ViT version and might not be applicable.
        # If 1D positional embedding is needed for the 100 windows, it should be added.
        # Remove or adapt ViT-specific initializations if not applicable to the CNN+Transformer structure.
        # h_num_patches = self.img_size[0] // self.patch_size[0] # 100
        # w_num_patches = self.img_size[1] // self.patch_size[1] # 50
        # pos_embed (1, 4000, 768)
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], h_num_patches, w_num_patches)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # Remove or adapt if PatchEmbed is not used.
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm---end---
        
        # This method will initialize weights for all submodules, including the selected embedding_module
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Original initialization logic
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        # Add specific initialization for Conv2d if needed, although xavier_uniform might cover it.
        elif isinstance(m, nn.Conv2d):
             torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
        # Note: AdaptiveLambdaEncoding has a learnable parameter C,
        # but nn.Parameter default initialization (from_tensor) is fine.

    # ---no need paych now---
    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)  e.g., [N, 3, 100, 50]
    #     x: (N, L, patch_size_h * patch_size_w * 3) e.g., [N, 5000, 1*1*3=3]
    #     """
    #     patch_size_h, patch_size_w = self.patch_embed.patch_size  # [1,1]
    #     assert imgs.shape[2] % patch_size_h == 0 and imgs.shape[3] % patch_size_w == 0
    #
    #     h = imgs.shape[2] // patch_size_h  # 100/1 = 100
    #     w = imgs.shape[3] // patch_size_w  # 50/1 = 50
    #     # [N, 3, 100, 1, 50, 1]
    #     x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, patch_size_h, w, patch_size_w))
    #
    #     # [N, 100, 50, 1, 1, 3]
    #     x = x.permute(0, 2, 4, 3, 5, 1)  # 重新排列维度以匹配期望的patch顺序
    #
    #     # [N, 100*50, 1*1*3] = [N, 5000, 3]
    #     x = x.reshape(shape=(imgs.shape[0], h * w, patch_size_h * patch_size_w * imgs.shape[1]))
    #     return x
    #
    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size_h * patch_size_w * 3) e.g., [N, 5000, 3]
    #     imgs: (N, 3, H, W) e.g., [N, 3, 100, 50]
    #     """
    #     patch_size_h, patch_size_w = self.patch_embed.patch_size # [1, 1]
    #     L = x.shape[1]  # L = 4000
    #     # 使用初始化时的 img_size 来确定 H 和 W
    #     original_H, original_W = self.img_size # (100, 50)
    #     in_chans = self.in_chans # 3
    #
    #     assert (original_H // patch_size_h) * (original_W // patch_size_w) == L
    #
    #     h = original_H // patch_size_h  # 100
    #     w = original_W // patch_size_w  # 50
    #     # [N, 100, 50, 1, 1, 3]
    #     x = x.reshape(shape=(x.shape[0], h, w, patch_size_h, patch_size_w, self.in_chans))
    #
    #     # permute()对数组或张量的维度进行重新排列。通过指定新的维度顺序，可以改变数组或张量的维度结构
    #     # [N, 3, 100, 1, 50, 1] - 调整 permute 顺序以匹配 patchify
    #     x = x.permute(0, 5, 1, 3, 2, 4)
    #
    #     # [N, 3, 100, 50]
    #     imgs = x.reshape(shape=(x.shape[0], in_chans, original_H, original_W))
    #
    #     return imgs
    # ---patch end---

    def forward_encoder(self, x, lambdas, deltas, masks, original_lengths):
        """
        x: [N, in_chans, 100, 50]. Channels: (μ, σ, error, [delta])  # Note: in_chans here can be 3 or 4.
           NOTE: Normalization of first 3 channels is handled INSIDE this method now.
        lambdas: [N, 100]
        deltas: [N, 100]
        masks: {'window': [N, 100], 'delta': [N, 100]}
        original_lengths: [N, 100] # Number of valid points in max_data dim for each window
        """
        N = x.shape[0]
        C = x.shape[1] # in_chans
        H = x.shape[2] # 100
        W = x.shape[3] # 50 (max_data)

        # --- Normalization ---
        # Only normalize the first 3 channels (μ, σ, error) regardless of C (in_chans)
        # Create a mask for normalization based on window_mask and original_lengths
        # window_mask is [N, 100]
        # original_lengths is [N, 100]

        # Expand mask and lengths for broadcasting against x_to_normalize [N, 3, 100, max_data]
        window_mask_expanded = masks['window'].unsqueeze(1).unsqueeze(-1) # [N, 1, 100, 1]

        # Create points mask based on original_lengths
        # Only need mask for the first 3 channels for normalization
        points_mask_norm = torch.zeros_like(x[:, :3, :, :], dtype=torch.bool, device=x.device) # [N, 3, 100, max_data]
        for i in range(N): # Iterate over batch size
             for j in range(H): # Iterate over windows (100)
                  length = int(original_lengths[i, j].item())
                  # Apply point mask based on original length, and window mask
                  # Note: Window mask should already filter out invalid windows, but apply again for safety
                  if masks['window'][i, j] > 0 and length > 0:
                       points_mask_norm[i, :, j, :length] = True # Mask applies to all 3 channels for these points

        # Cast window_mask_expanded to boolean before bitwise AND
        effective_norm_mask = window_mask_expanded.bool() & points_mask_norm # [N, 3, 100, max_data]

        # Create the tensor that will be passed to the embedding module
        x_processed = x.clone() # Clone to avoid modifying input 'x'

        # Apply normalization only to the masked elements in the first 3 channels
        epsilon = 1e-6
        # Use indices from effective_norm_mask to select corresponding means and stds from buffer
        # Use means and stds from buffer, ensure they are on the correct device
        current_train_means = self.train_means.to(x.device) # [3]
        current_train_stds = self.train_stds.to(x.device)   # [3]

        # Broadcasting means (3,) and stds (3,) will apply correctly across [N, 3, 100, max_data] mask
        # No need for nonzero indexing if applying directly with boolean mask
        # x_processed[:, :3, :, :][effective_norm_mask] = (x_processed[:, :3, :, :][effective_norm_mask] - current_train_means[effective_norm_mask.nonzero(as_tuple=True)[1]]) / (current_train_stds[effective_norm_mask.nonzero(as_tuple=True)[1]] + epsilon)

        # A cleaner way to apply normalization with broadcasting:
        # Apply normalization only to the first 3 channels, and the mask zeros out the effect elsewhere
        # Use expanded means and stds for broadcasting during normalization: [1, 3, 1, 1]
        means_broadcast = current_train_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, 3, 1, 1]
        stds_broadcast = current_train_stds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, 3, 1, 1]

        # Normalize the first 3 channels
        x_processed[:, :3, :, :] = (x_processed[:, :3, :, :] - means_broadcast) / (stds_broadcast + epsilon)

        # Zero out elements in the first 3 channels that were not in the original data using the expanded *normalization* mask
        # effective_norm_mask is [N, 3, 100, max_data]
        x_processed[:, :3, :, :][~effective_norm_mask] = 0.0 # Set padded/masked values to 0 after normalization

        # If in_chans is 4, the 4th channel (delta) is not normalized and is kept as is in x_processed
        # The embedding module will receive x_processed which has normalized channels 0-2 and original channel 3 (if it exists)

        # --- Embedding ---
        # Pass the processed data (normalized first 3 channels, plus delta if C=4),
        # lambda, delta, masks, lengths to the embedding module.
        # The embedding module handles how to combine these based on its strategy and in_chans.
        lambda_emb, feat_emb = self.embedding_module(x_processed, lambdas, deltas, masks, original_lengths)
        # lambda_emb shape: [N, 100, embed_dim]
        # feat_emb shape: [N, 100, embed_dim]

        # --- Feature Fusion ---
        # Add lambda embedding and feature embedding
        x = lambda_emb + feat_emb  # [N, 100, embed_dim]

        # --- Transformer Blocks ---
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x  # [N, 100, embed_dim]

        # Modified forward method to accept target data and return losses and pred_dGs_per_window
    # Modified forward method signature remains the same for external users
    def forward(self,
                imgs: torch.Tensor, # [N, C, 100, 50]. Channels: (μ, σ, error, [delta]) - Raw input
                lambdas: torch.Tensor, # [N, 100]
                deltas: torch.Tensor, # [N, 100]
                masks: Dict[str, torch.Tensor], # {'window': [N, 100], 'delta': [N, 100]}
                original_lengths: torch.Tensor, # [N, 100] - Number of valid points in max_data dim for each window
                target_total_dg: torch.Tensor, # Add target_total_dg [N] (kcal/mol)
                original_window_dGs_list: list[torch.Tensor], # Add original_window_dGs_list (list of variable length tensors, kcal/mol)
                original_dlambda_list: list[float], # Add original_dlambda_list (list of floats)
                reduction: str = 'mean', # Add reduction for the final loss
                # --- MODIFIED: Receive lists from collate_fn ---
                all_original_mu_targets_list: list[torch.Tensor] = None,
                all_original_sigma_targets_list: list[torch.Tensor] = None,
                all_original_100_grid_indices_list: list[torch.Tensor] = None
                # --- END MODIFIED ---
                ):
        # 编码 - Pass original_lengths to forward_encoder
        # forward_encoder handles input normalization and embedding selection
        latent = self.forward_encoder(imgs, lambdas, deltas, masks, original_lengths)  # [N, 100, embed_dim]

        # 投影到3个特征通道 (预测的是 μ, σ, error)
        # According to Option A, the projection head outputs PREDICTIONS OF NORMALIZED features
        pred_features_normalized = self.projection_head(latent)  # [N, 100, 3]

        # 转换为 [N, 3, 100]
        pred_normalized = pred_features_normalized.permute(0, 2, 1)  # [N, 3, 100]

        # --- Option A: Denormalize Model Output to get Predicted (μ, σ, error) ---
        # pred_normalized is the model's prediction, assumed to be normalized here.
        # Denormalization formula: data = normalized_data * std + mean
        # Ensure means and stds are on the correct device and correctly broadcast for shape [N, 3, 100]
        current_train_means = self.train_means.to(pred_normalized.device) # [3]
        current_train_stds = self.train_stds.to(pred_normalized.device)   # [3]

        # Reshape means/stds from [3] to [1, 1, 3] for broadcasting with [N, 3, 100] pred_normalized
        means_broadcast = current_train_means.unsqueeze(0).unsqueeze(-1) # [1, 3, 1]
        stds_broadcast = current_train_stds.unsqueeze(0).unsqueeze(-1)   # [1, 3, 1]

        # Apply denormalization to get predicted (μ, σ, error) in the original scale
        pred = pred_normalized * stds_broadcast + means_broadcast # [N, 3, 100] (DENORMALIZED μ, σ, error)

        # --- Option B (Commented Out): Assume Projection Head Outputs Unnormalized Features ---
        # To test Option B: uncomment the line below and comment out the denormalization block above.
        # pred = pred_features_normalized # In this case, pred_features_normalized is assumed to be the unnormalized prediction already
        # -------------------------------------------------------------------------------------

        # Calculate predicted dGs per window from the DENORMALIZED prediction (pred)
        # Permute pred to [N, 3, 100] for easier channel access, if needed for calculation
        # Calculation below uses [N, 100, 3] shape which is fine

        # Assumed predicted features are μ (channel 0), σ (channel 1), error (channel 2) in the last dimension
        mu_pred = pred[:, 0, :] # [N, 100]
        sigma_pred = pred[:, 1, :] # [N, 100]
        error_pred = pred[:, 2, :] # [N, 100]

        # Ensure predicted sigma (channel 1) is non-negative. Sigma must be >= 0.
        # Apply relu after denormalization to ensure sigma is physical
        sigma_pred = F.softplus(sigma_pred) # Or use softplus for smoothness

        # Convert predicted sigma (σ) to variance (σ²)
        sigma_sq_pred = sigma_pred ** 2 # [N, 100]

        # Calculate dG_n for each window (kbt=0.592 assumed here)
        kbt = 0.592 # Make sure KBT is consistent
        # Use sigma_sq in the dG_n formula
        # dG_n = μ - σ²/2 + error  (per window sum before kbt)
        dG_n_per_window = mu_pred - (sigma_sq_pred / 2) + error_pred # [N, 100] (kbt=1 scale sum)

        # Total dG for the system is the sum of dG_n for all 100 windows * kbt
        # pred_dGs_per_window below is the value for dG_n * kbt for each window, [N, 100]
        pred_dGs_per_window = dG_n_per_window * kbt # [N, 100] (kbt=0.592 scale, kcal/mol)

        # --- Calculate Smoothness Loss ---
        # Calculate second derivative approximation for mu and sigma along the window dimension (dim=1)
        # Shape is [N, 100]
        # We can calculate this for indices 1 to 98
        mu_second_deriv = mu_pred[:, 2:] - 2 * mu_pred[:, 1:-1] + mu_pred[:, :-2] # [N, 98]
        sigma_second_deriv = sigma_pred[:, 2:] - 2 * sigma_pred[:, 1:-1] + sigma_pred[:, :-2] # [N, 98]

        # Calculate squared second derivative and mean across the 98 points per sample
        mu_smoothness_loss_per_sample = torch.mean(mu_second_deriv ** 2, dim=1) # [N]
        sigma_smoothness_loss_per_sample = torch.mean(sigma_second_deriv ** 2, dim=1) # [N]

        # Combine mu and sigma smoothness losses per sample
        # We can add them directly or use separate weights if needed
        # For now, sum them
        smoothness_loss_per_sample = mu_smoothness_loss_per_sample + sigma_smoothness_loss_per_sample # [N]

        # Apply the overall smoothness loss weight
        weighted_smoothness_loss_per_sample = self.smoothness_loss_weight * smoothness_loss_per_sample # [N]

        # Calculate losses using the dedicated method
        # Pass the predicted dG_n (kbt=0.592 scale) and target data
        # Pass the calculated weighted_smoothness_loss_per_sample to calculate_loss
        losses_dict = self.calculate_loss(
            pred=pred, # Pass DENORMALIZED prediction (μ, σ, error) [N, 3, 100]
            pred_dGs_per_window=pred_dGs_per_window, # Pass dG_n * kbt (kcal/mol) [N, 100]
            target_total_dg=target_total_dg, # Target total dG (kcal/mol) [N]
            original_window_dGs_list=original_window_dGs_list, # Target window dGs (list of variable length tensors, kcal/mol)
            original_dlambda_list=original_dlambda_list, # Target dlambda (list of floats)
            smoothness_loss_per_sample=weighted_smoothness_loss_per_sample, # Pass weighted smoothness loss per sample [N]
            reduction=reduction, # Pass reduction to calculate_loss
            processed_lambdas=lambdas, # Pass processed_lambdas [N, 100]
            processed_masks=masks, # Pass processed_masks (Dict)
            # --- MODIFIED: Pass the lists using the correct parameter names ---
            all_original_mu_targets_list=all_original_mu_targets_list,
            all_original_sigma_targets_list=all_original_sigma_targets_list,
            all_original_100_grid_indices_list=all_original_100_grid_indices_list,
            # --- END MODIFIED ---
        )

        # Return the dictionary of losses, predicted dGs per window (kcal/mol), predicted total dg (kcal/mol), and target total dG (kcal/mol)
        return (losses_dict,
                pred_dGs_per_window, # dG_n * kbt per window (kcal/mol) [N, 100]
                losses_dict['predicted_total_dg'], # Total dG (kcal/mol) [N]
                losses_dict['target_total_dg'] # Target Total dG (kcal/mol) [N]
               )

    # calculate_loss method remains largely the same, operating on predicted dGs and targets
    # Modified calculate_loss method to accept smoothness loss
    def calculate_loss(self,
                       pred: torch.Tensor, # Model output [N, 3, 100] (DENORMALIZED μ, σ, error)
                       pred_dGs_per_window: torch.Tensor, # Predicted dG_n * kbt per window [N, 100] (kcal/mol)
                       target_total_dg: torch.Tensor, # Target total dG [N] (kcal/mol)
                       original_window_dGs_list: list[torch.Tensor], # Target window dGs (variable length) (kcal/mol)
                       original_dlambda_list: list[float], # Original delta_lambda (list)
                       smoothness_loss_per_sample: torch.Tensor, # Add weighted smoothness loss per sample [N]
                       reduction: str = 'mean', # Reduction for the final combined loss
                       processed_lambdas: torch.Tensor = None, # [N, 100] (Standard lambda grid)
                       processed_masks: Dict[str, torch.Tensor] = None, # {'window': [N, 100], ...}
                       # --- MODIFIED: Receive lists for all original window targets and indices ---
                       all_original_mu_targets_list: list[torch.Tensor] = None, # List of [num_original_windows_total_i]
                       all_original_sigma_targets_list: list[torch.Tensor] = None, # List of [num_original_windows_total_i]
                       all_original_100_grid_indices_list: list[torch.Tensor] = None # List of [num_original_windows_total_i]
                       # --- END MODIFIED ---
                       ):
        """
        Calculate the combined loss based on total dG and aggregated window dGs.
        Includes smoothness and supervised feature regularization losses.
        Calculates feature loss by supervising model's 100-grid predictions
        at original window locations using targets from ALL original windows.

        Args:
            pred (torch.Tensor): Model output [N, 3, 100] (DENORMALIZED μ, σ, error).
            pred_dGs_per_window (torch.Tensor): Predicted dG_n * kbt per window [N, 100] (kcal/mol).
            target_total_dg (torch.Tensor): Target total dG for the system [N] (kcal/mol).
            original_window_dGs_list (list[torch.Tensor]): Target window dGs for the original system (list of variable length tensors, kcal/mol).
            original_dlambda_list (list[float]): Original delta_lambda (list).
            smoothness_loss_per_sample (torch.Tensor): Weighted smoothness loss per sample [N].
            reduction (str): Specify the reduction method: 'none' | 'mean' | 'sum'. Default: 'mean'.
            processed_lambdas (torch.Tensor): Standard 100-grid lambda values for each sample [N, 100].
            processed_masks (Dict[str, torch.Tensor]): Masks from processor {'window': [N, 100], ...}.
            all_original_mu_targets_list (list[torch.Tensor]): List of mu targets for ALL original windows of each sample.
            all_original_sigma_targets_list (list[torch.Tensor]): List of sigma targets for ALL original windows of each sample.
            all_original_100_grid_indices_list (list[torch.Tensor]): List of 100-grid indices for ALL original windows of each sample.


        Returns:
            dict: A dictionary containing 'total_dg_loss', 'agg_dg_loss', 'smoothness_loss',
                  'combined_loss', 'predicted_total_dg', and 'target_total_dg'.
                  'smoothness_loss' here is the per-sample weighted smoothness loss (before final reduction).
        """
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}")

        N = pred.shape[0] # Batch size

        # --- Calculate Original Total dG Loss (MSE) ---
        # pred_dGs_per_window is already calculated from denormalized pred
        # Sum predicted dG_n * kbt over 100 windows to get predicted total dG (kcal/mol)
        predicted_total_dg = torch.sum(pred_dGs_per_window, dim=1) # [N]

        # Calculate MSE loss for total dG, per sample
        total_dg_loss_per_sample = (predicted_total_dg - target_total_dg) ** 2 # [N]


        # --- Calculate New Aggregated Window dG Loss ---
        # Use the predicted dGs per window (pred_dGs_per_window, [N, 100], kcal/mol)
        # and the original target window dGs (original_window_dGs_list, list of [num_original_windows], kcal/mol)
        # and original delta_lambda (original_dlambda_list, list of floats)
        # dg_aggregation_loss_v2 with reduction='none' returns mean loss for each sample [N]
        agg_dg_loss_per_sample = dg_aggregation_loss_v2(
            pred_dGs=pred_dGs_per_window, # Pass the dG_n * kbt predictions (kcal/mol)
            original_window_dGs_list=original_window_dGs_list, # Targets in kcal/mol
            original_dlambda_list=original_dlambda_list,
            reduction='none' # Return mean loss for each sample [N]
        ) # [N]

        # --- Calculate Feature Loss (Supervised) ---
        # Use all_original_mu_targets_list, all_original_sigma_targets_list, all_original_100_grid_indices_list (lists of variable length tensors)
        # To supervise pred [N, 3, 100] at specific indices.

        # Check if targets are provided
        if all_original_mu_targets_list is None or all_original_sigma_targets_list is None or all_original_100_grid_indices_list is None or \
           len(all_original_mu_targets_list) != N or len(all_original_sigma_targets_list) != N or len(all_original_100_grid_indices_list) != N:

            # Feature loss is 0 if targets are not provided or lengths mismatch batch size
            # print("Warning: All original window targets (mu, sigma, indices) lists not provided or list length mismatch batch size. Feature loss is 0.") # Avoid excessive printing
            feature_loss_per_sample = torch.zeros(N, device=pred.device) # [N]

        else:
            # Get predicted mu and sigma from model's 100 windows
            # pred is [N, 3, 100] (DENORMALIZED)
            mu_pred_01_all_grid = pred[:, 0, :]  # [N, 100]
            sigma_pred_01_all_grid = pred[:, 1, :]  # [N, 100]
            # Ensure sigma is non-negative
            sigma_pred_01_all_grid = F.softplus(sigma_pred_01_all_grid)  # torch.reLU --> softplus

            feature_loss_per_sample = torch.zeros(N, device=pred.device) # Initialize per-sample feature loss [N]

            # Iterate over each sample in the batch
            for i in range(N):
                original_mu_targets_i = all_original_mu_targets_list[i].to(pred.device) # Targets for all original windows of this sample
                original_sigma_targets_i = all_original_sigma_targets_list[i].to(pred.device)
                original_100_grid_indices = all_original_100_grid_indices_list[i].to(pred.device) # Indices in the 100 grid

                # Check if there are any original windows for this sample
                if original_mu_targets_i.numel() == 0: # Check if original targets list is empty
                    # No original windows for this sample, feature loss is 0 for this sample
                    feature_loss_per_sample[i] = 0.0
                    continue # Move to the next sample

                # Get predicted mu and sigma for the current sample across all 100 grid points
                pred_mu_i = mu_pred_01_all_grid[i]     # [100]
                pred_sigma_i = sigma_pred_01_all_grid[i] # [100]

                # Determine which of the 100 predicted windows have a corresponding original target
                # The original targets correspond to lambdas 0.00, 0.02, 0.04, ...
                # For a predicted lambda at index j (0..99), its corresponding original target index is floor(j/2)
                
                # Create indices for the 100-grid (0 to 99)
                indices_100_grid = torch.arange(100, device=pred.device)

                # Map each 100-grid index to its potential original target index (0-based for original_mu_targets)
                # e.g., 0->0, 1->0, 2->1, 3->1, ...
                mapped_original_target_indices = indices_100_grid // 2

                # Create a mask to select only those 100-grid predictions that have a valid original target
                # A target is valid if its mapped_original_target_index is less than the number of available original targets
                valid_target_mask = mapped_original_target_indices < original_mu_targets_i.numel()

                # If no valid targets based on this mapping, loss is 0 for this sample
                if not valid_target_mask.any():
                    feature_loss_per_sample[i] = 0.0
                    continue

                # Select the predictions from the 100-grid that have valid targets
                constrained_pred_mu = pred_mu_i[valid_target_mask]
                constrained_pred_sigma = pred_sigma_i[valid_target_mask]

                # Select the corresponding original target indices
                effective_original_indices = mapped_original_target_indices[valid_target_mask]

                # Use these effective_original_indices to get the actual target values
                constrained_target_mu = torch.index_select(original_mu_targets_i, 0, effective_original_indices)
                constrained_target_sigma = torch.index_select(original_sigma_targets_i, 0, effective_original_indices)

                # Apply the 2x relationship to the predicted values
                # This assumes the 0.01-step predictions are "half" the 0.02-step targets at the corresponding lambda.
                scaled_predicted_mu = 2 * constrained_pred_mu
                scaled_predicted_sigma = 2 * constrained_pred_sigma

                # Calculate MSE for mu and sigma feature losses for this sample, averaged over its constrained windows
                mu_feature_loss_sample = torch.mean((scaled_predicted_mu - constrained_target_mu) ** 2)
                sigma_feature_loss_sample = torch.mean((scaled_predicted_sigma - constrained_target_sigma) ** 2)

                # Total Feature Loss for this sample
                feature_loss_per_sample[i] = mu_feature_loss_sample + sigma_feature_loss_sample

        # Apply the overall feature loss weight
        weighted_feature_loss_per_sample = self.feature_loss_weight * feature_loss_per_sample # [N]

        # Combine losses per sample with weights
        combined_loss_per_sample = self.total_dg_loss_weight * total_dg_loss_per_sample + \
                                   self.agg_dg_loss_weight * agg_dg_loss_per_sample + \
                                   smoothness_loss_per_sample + \
                                   weighted_feature_loss_per_sample # Add weighted feature loss # [N]

        # Apply final reduction
        if reduction == 'none':
            final_combined_loss = combined_loss_per_sample
            # For 'none', the return values should also reflect 'none' reduction (per sample)
            total_dg_loss_output = total_dg_loss_per_sample
            agg_dg_loss_output = agg_dg_loss_per_sample
            smoothness_loss_output = smoothness_loss_per_sample
            feature_loss_output = feature_loss_per_sample # Return per-sample feature loss (before weight)
        elif reduction == 'mean':
            # Ensure handling empty batch case
            if combined_loss_per_sample.numel() > 0:
                final_combined_loss = torch.mean(combined_loss_per_sample)
                total_dg_loss_output = torch.mean(total_dg_loss_per_sample)
                agg_dg_loss_output = torch.mean(agg_dg_loss_per_sample)
                smoothness_loss_output = torch.mean(smoothness_loss_per_sample)
                feature_loss_output = torch.mean(feature_loss_per_sample) # Return mean of per-sample feature loss (before weight)
            else: # Handle empty batch explicitly
                final_combined_loss = torch.tensor(0.0, device=pred.device)
                total_dg_loss_output = torch.tensor(0.0, device=pred.device)
                agg_dg_loss_output = torch.tensor(0.0, device=pred.device)
                smoothness_loss_output = torch.tensor(0.0, device=pred.device)
                feature_loss_output = torch.tensor(0.0, device=pred.device)

        elif reduction == 'sum':
             # Ensure handling empty batch case
             if combined_loss_per_sample.numel() > 0:
                 final_combined_loss = torch.sum(combined_loss_per_sample)
                 total_dg_loss_output = torch.sum(total_dg_loss_per_sample)
                 agg_dg_loss_output = torch.sum(agg_dg_loss_per_sample)
                 smoothness_loss_output = torch.sum(smoothness_loss_per_sample)
                 feature_loss_output = torch.sum(feature_loss_per_sample) # Return sum of per-sample feature loss (before weight)
             else: # Handle empty batch explicitly
                 final_combined_loss = torch.tensor(0.0, device=pred.device)
                 total_dg_loss_output = torch.tensor(0.0, device=pred.device)
                 agg_dg_loss_output = torch.tensor(0.0, device=pred.device)
                 smoothness_loss_output = torch.tensor(0.0, device=pred.device)
                 feature_loss_output = torch.tensor(0.0, device=pred.device)
        # else is handled by the initial check


        return {
            'total_dg_loss': total_dg_loss_output,
            'agg_dg_loss': agg_dg_loss_output,
            'smoothness_loss': smoothness_loss_output,
            'feature_loss': feature_loss_output, # Add feature loss to the output dictionary
            'combined_loss': final_combined_loss, # Return the loss with specified reduction
            'predicted_total_dg': predicted_total_dg, # Add predicted_total_dg to the dict (kcal/mol) [N]
            'target_total_dg': target_total_dg # Add target_total_dg to the dict (kcal/mol) [N]
        }


# -------------------------------------------------

def enc_cnn_chans4(**kwargs):
    # Get train_means and train_stds from kwargs
    train_means = kwargs.pop('train_means', None)
    train_stds = kwargs.pop('train_stds', None)
    # Get in_chans from kwargs, default to 3
    in_chans = kwargs.pop('in_chans', 4)  # Allow passing in_chans via kwargs
    # Get embedding_strategy_type for in_chans=4
    embedding_strategy_type = kwargs.pop('embedding_strategy_type', 'mlp')  # 'cnn', 'flatten_linear', 'mlp'
    # Get smoothness_loss_weight
    smoothness_loss_weight = kwargs.pop('smoothness_loss_weight', 0.0)
    # Get feature_loss_weight
    feature_loss_weight = kwargs.pop('feature_loss_weight', 1.0) # Default feature loss weight to 0
    
    model = MaskedAutoencoderViT(
        img_size=(100, 50),
        in_chans=in_chans,  # Use the potentially adjusted in_chans
        embed_dim=384,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        target_output_windows=100,  # Explicitly pass target window count
        total_dg_loss_weight=kwargs.pop('total_dg_loss_weight', 1),
        agg_dg_loss_weight=kwargs.pop('agg_dg_loss_weight', 0.1),
        smoothness_loss_weight=smoothness_loss_weight,  # Pass smoothness weight
        feature_loss_weight=feature_loss_weight,  # Pass feature weight
        train_means=train_means, # Pass train_means to constructor
        train_stds=train_stds, # Pass train_stds to constructor
        embedding_strategy_type=embedding_strategy_type,  # Pass embedding_strategy_type
        # smoothness_loss_weight=smoothness_loss_weight, # 传递平滑损失权重
        **kwargs)
    return model


def enc_cnn_chans3(**kwargs):
    # Get train_means and train_stds from kwargs
    train_means = kwargs.pop('train_means', None)
    train_stds = kwargs.pop('train_stds', None)
    in_chans = kwargs.pop('in_chans', 3)
    smoothness_loss_weight = kwargs.pop('smoothness_loss_weight', 0.0)
    feature_loss_weight = kwargs.pop('feature_loss_weight', 1.0)  # Default feature loss weight to 0
    model = MaskedAutoencoderViT(
        img_size=(100, 50),
        in_chans=in_chans,  # Use the potentially adjusted in_chans
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        target_output_windows=100,  # Explicitly pass target window count
        total_dg_loss_weight=kwargs.pop('total_dg_loss_weight', 1),
        agg_dg_loss_weight=kwargs.pop('agg_dg_loss_weight', 0.1),
        smoothness_loss_weight=smoothness_loss_weight,  # Pass smoothness weight
        feature_loss_weight=feature_loss_weight,  # Pass feature weight
        train_means=train_means,  # Pass train_means to constructor
        train_stds=train_stds,  # Pass train_stds to constructor
        **kwargs)
    return model


if __name__ == '__main__':
    #
    # # 1. 首先设置环境变量（必须在任何CUDA操作之前）调整显存分配策略 限制 PyTorch 分配显存时的 最大块大小，减少碎片化。
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # 单位MB，建议32-256范围测试
    #
    # # 2. 然后限制PyTorch显存预留（紧接环境变量之后）
    # torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # 限制为60%显存
    #
    # # 3. 最后清理缓存（在所有设置完成后）
    # torch.cuda.empty_cache()  # 清理残留缓存

    # -------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 参数设置
    batch_size = 2
    processor_max_data = 50
    # Set this to True to test with 4 channels. Note: LambdaDataProcessor needs update for this.
    processor_include_delta = True
    in_chans_simulated = 3 if not processor_include_delta else 4 # Determine raw data channels for simulation

    # Add simulated embedding strategy type for in_chans=4 testing (default 'cnn' for now)
    # This will be used when in_chans_simulated is 4.
    embedding_strategy_type_simulated = 'cnn' # Can be 'cnn', 'flatten_linear', 'mlp' (when implemented)

    # --- NEW: Simulate per_lambda_max_points for test data generation ---
    simulated_per_lambda_max_points = 20 # Each lambda window will have at most 20 data points in simulation
    print(f"Simulating each lambda window to contain at most {simulated_per_lambda_max_points} data points.")
    # --- END NEW ---

    # --- IMPORTANT ---
    # The LambdaDataProcessor definition in THIS file still needs to be updated
    # to match the one in test_tool/util/test_lambda_emb_dataset.py which supports
    # include_delta_in_data.
    # For this test to run with in_chans_simulated=4, please update the
    # LambdaDataProcessor class definition above with the one from the other file.
    # The current code will assert fail if processor_include_delta is True
    # because the processor in this file only outputs 3 channels.
    # --- IMPORTANT ---
    processor = LambdaDataProcessor(min_delta=0.01, max_windows=100, max_data=processor_max_data, include_delta_in_data=processor_include_delta) # Uncomment after updating processor class
    print(f"Processor configured with max_windows={processor.max_windows}, max_data={processor.max_data}, include_delta_in_data={processor_include_delta}")

    # 2. 生成测试数据 (需要模拟原始窗口 dGs 和 dlambda)
    print("\nGenerating test data...")
    raw_batch_data = []       # List[List[Tensor[3 or 4, num_points]]] (Channels: μ, σ, error, [delta])
    original_lambdas = []     # List[List[float]]
    original_deltas = []      # List[List[float]]
    original_data_lengths = []  # List[List[int]]
    # New: Simulate original system data for target losses
    original_window_dGs_list_sim = [] # List[Tensor] (kcal/mol)
    original_dlambda_list_sim = [] # List[float]
    target_total_dg_list_sim = [] # List[float] (kcal/mol)

    all_possible_lambdas = np.round(np.arange(0, 1.01, 0.01), 2)

    # 为批次中的每个样本生成数据
    for i in range(batch_size):
        # 确定当前样本的lambda窗口数量
        lambda_count = np.random.randint(10, 51)  # 10到50个窗口

        # 生成lambdas和deltas
        lambdas = sorted(np.random.choice(all_possible_lambdas, size=lambda_count, replace=False).tolist())
        deltas = np.round(np.random.uniform(0.01, 0.1, lambda_count), 2).tolist() # Simulate deltas for each window
        original_lambdas.append(lambdas)
        original_deltas.append(deltas)

        # 为每个lambda窗口生成数据并计算其 dG_n
        sample_data_lengths = []
        sample_raw_data = []
        sample_window_dGs_n = [] # Collect dG_n for this sample's original windows (kbt=1 scale)

        kbt = 0.592 # Ensure KBT consistency
        sample_total_dg_sum = 0.0 # Sum for calculating total dG in kcal/mol

        for j in range(lambda_count):
            data_length = np.random.randint(4, processor.max_data+1)
            sample_data_lengths.append(data_length)
            
            # Simulate raw data: μ, σ, error (3 channels)
            # Shape [3, data_length]
            window_data_3chan = torch.randn(3, data_length, dtype=torch.float32)
            # Ensure sigma (channel 1) is non-negative for simulation
            window_data_3chan[1, :] = torch.abs(window_data_3chan[1, :]) # Assuming channel 1 is sigma

            # If processor_include_delta is True, add a 4th channel
            if processor_include_delta:
                 # Add delta as the 4th channel, broadcast to data_length
                 current_delta_tensor = torch.full((1, data_length), deltas[j], dtype=torch.float32)
                 window_data = torch.cat([window_data_3chan, current_delta_tensor], dim=0) # Shape [4, data_length]
            else:
                 window_data = window_data_3chan # Shape [3, data_length]

            # --- NEW: Apply truncation based on simulated_per_lambda_max_points to generated data ---
            if window_data.shape[1] > simulated_per_lambda_max_points:
                window_data = window_data[:, :simulated_per_lambda_max_points]
            # Update data_length to reflect the truncation
            data_length = window_data.shape[1]
            # --- END NEW ---

            sample_raw_data.append(window_data)

            # Calculate dG_n for this window (features [mu, sigma_sq, error, ...] in dim 0)
            # Use the last point for dG calculation as per calculate_total_dg logic
            # Assuming features are [mu, sigma, error] in channels 0, 1, 2
            mu_j = window_data[0, -1]
            sigma_j = window_data[1, -1] # Use simulated sigma value
            error_j = window_data[2, -1]

            # Calculate variance (σ²) from simulated sigma (σ)
            sigma_sq_j = sigma_j ** 2

            # Calculate dG_n (kbt=1 scale)
            dG_n_j = (mu_j - (sigma_sq_j / 2.0) + error_j) * kbt
            sample_window_dGs_n.append(dG_n_j.item())

            # Sum up dG_n * kbt for total dG in kcal/mol
            sample_total_dg_sum += dG_n_j.item()

        original_data_lengths.append(sample_data_lengths)
        raw_batch_data.append(sample_raw_data)

        # Store simulated original system targets
        original_window_dGs_list_sim.append(torch.tensor(sample_window_dGs_n, dtype=torch.float32)) # List of Tensors
        # Assuming delta is consistent for a system, use the first delta as the representative dlambda
        original_dlambda_list_sim.append(deltas[0] if deltas else 0.01) # List of floats
        target_total_dg_list_sim.append(sample_total_dg_sum) # List of floats (kcal/mol)

        print(f"  Sample {i}: Generated {lambda_count} windows, simulated total dG: {sample_total_dg_sum:.4f}")


    # 3. 数据预处理 (for model input)
    print("\nProcessing data...")
    # The processor outputs data aligned to 100 windows
    # Note: The processor's output 'data' will have the same channels as raw_batch_data (μ, σ, error, [delta])
    # Note: The processor in THIS file currently outputs 3 channels ('data' key).
    # If in_chans_simulated is 4, this will cause a mismatch.
    # We need to update the LambdaDataProcessor in THIS file to match the one in test_tool/util/test_lambda_emb_dataset.py.
    # For now, let's proceed assuming the processor provides the expected in_chans_simulated.
    # We'll add the processor update as a separate task.

    processed_data_for_model = processor.process(raw_batch_data, original_lambdas, original_deltas, original_data_lengths)

    # Check output channels of the processor
    # This check will fail if LambdaDataProcessor in this file is not updated for 4 channels.
    # We will update the processor in the next steps.
    # if processed_data_for_model['data'].shape[1] != in_chans_simulated:
    #      print(f"Error: Processor output channels ({processed_data_for_model['data'].shape[1]}) mismatch simulated input channels ({in_chans_simulated}).")
         # Adjust in_chans for model initialization based on processor output if needed, or fix processor/simulation
         # Assuming processor output channels match the raw input channels simulated
    #      pass # Proceed, assuming they match
    # Let's add an assertion here instead, as the goal is to make them match
    expected_processor_out_chans = processed_data_for_model['data'].shape[1]
    assert expected_processor_out_chans == in_chans_simulated, \
         f"Processor output channels ({expected_processor_out_chans}) mismatch simulated input channels ({in_chans_simulated}). Update LambdaDataProcessor in this file."


    # 检查输出
    aligned_data_tensor = processed_data_for_model['data']
    output_masks = processed_data_for_model['masks']
    output_original_lengths = processed_data_for_model['original_lengths']

    print(f"Processed data shape (for model input): {aligned_data_tensor.shape}") # Should match [N, C, 100, max_data] where C is 3 or 4
    print(f"Mask shapes: window={output_masks['window'].shape}, delta={output_masks['delta'].shape}")
    print(f"Original Lengths shape: {output_original_lengths.shape}")

    # --- NEW: Check and print sampled lambdas from a processed subset ---
    if batch_size > 0:
        # Assuming we want to check the first sample in the batch
        sample_idx_to_check = 0

        # aligned_lambdas tensor 包含了填充后的 100 个 lambda 值
        # Note: processed_data_for_model['lambdas'] is [N, 100]. Pick one sample.
        aligned_lambdas_for_sample = processed_data_for_model['lambdas'][sample_idx_to_check] # Shape [100]
        # window_mask 标记了哪些位置是原始数据填充进来的有效窗口
        window_mask_for_sample = processed_data_for_model['masks']['window'][sample_idx_to_check] # Shape [100]

        # 找出 mask 中值为 1 的索引，这些索引对应着原始数据填充的位置
        original_subset_lambdas_indices = torch.where(window_mask_for_sample > 0)[0]
        # 使用这些索引从 aligned_lambdas 中提取出原始 lambda 值
        original_subset_lambdas = aligned_lambdas_for_sample[original_subset_lambdas_indices]

        print(f"\nSampled Lambdas for batch sample {sample_idx_to_check} (from processor output):")
        # 将 Tensor 转换为 NumPy 数组以便打印，并格式化输出
        print(np.round(original_subset_lambdas.cpu().numpy(), 4))
        print(f"Number of sampled windows: {len(original_subset_lambdas)}")
    # --- END NEW ---

    # Simulate training statistics here for testing purposes
    # These stats should be for μ, σ, error (3 channels)
    # In real training, these would come from the calculation in main.py
    simulated_train_means = torch.randn(3).to(device) # Random means for μ, σ, error
    simulated_train_stds = torch.rand(3).to(device) + 0.1 # Random stds > 0 for μ, σ, error
    # Ensure stds are not zero
    simulated_train_stds[simulated_train_stds < 1e-6] = 1.0
    # Ensure means/stds are on the same device as the data (will be moved to CPU buffer in model init)
    simulated_train_means = simulated_train_means.to(device)
    simulated_train_stds = simulated_train_stds.to(device)

    print(f"\nSimulated train_means for testing: {simulated_train_means.cpu().numpy()}")
    print(f"Simulated train_stds for testing:  {simulated_train_stds.cpu().numpy()}")

    # Convert simulated target lists to tensors where needed by calculate_loss
    # original_window_dGs_list_sim stays as list[Tensor]
    # original_dlambda_list_sim stays as list[float]
    target_total_dg_tensor_sim = torch.tensor(target_total_dg_list_sim, dtype=torch.float32).to(device)


    # 4. 初始化模型 (pass loss weights, simulated stats, and smoothness weight)
    print("\nInitializing model...")
    # Instantiate model with default or specified loss weights and simulated stats
    # Add smoothness_loss_weight here for testing
    test_smoothness_weight = 0.1 # Set a non-zero weight for testing
    print(f"Initializing model with smoothness_loss_weight={test_smoothness_weight}")

    model = enc_cnn_chans4(
        in_chans=in_chans_simulated, # Pass the actual number of input channels (3 or 4)
        total_dg_loss_weight=1.0, # Example weights
        agg_dg_loss_weight=1.0,
        train_means=simulated_train_means, # Pass simulated means
        train_stds=simulated_train_stds, # Pass simulated stds
        embedding_strategy_type=embedding_strategy_type_simulated, # Pass simulated strategy type
        smoothness_loss_weight=test_smoothness_weight, # Pass the test smoothness weight
    ).to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {total_params:,}")
    print(f"参数显存: {total_params * 4 / 1024 ** 2:.2f} MB")

    # Verify stored stats in the model (will be on CPU buffer)
    print(f"\nStats stored in model buffer: train_means={model.train_means.cpu().numpy()}, train_stds={model.train_stds.cpu().numpy()}")


    # 5. 前向传播测试 - 直接返回损失和 predicted dGs per window
    print("\nPerforming forward pass (calculating losses internally and returning preds/targets)...")

    # Prepare all inputs for the forward pass (model inputs + target data)
    input_data = aligned_data_tensor.to(device)
    masks_input = {k: v.to(device) for k, v in output_masks.items()}
    original_lengths_input = output_original_lengths.to(device) # Add original_lengths to inputs

    # Call model - it now returns the losses dictionary, pred_dGs_per_window, predicted_total_dg, and target_total_dg
    # Pass original_lengths to the model forward method
    # This call will fail if in_chans_simulated is 4 and EmbeddingStrategy4Chans is not fully implemented.
    losses, pred_dGs_per_window, predicted_total_dg, target_total_dg = model( # <-- Modified to capture four return values
        input_data, # Model input 1 [N, C, 100, max_data] (C is in_chans_simulated)
        processed_data_for_model['lambdas'].to(device), # Model input 2 [N, 100]
        processed_data_for_model['deltas'].to(device), # Model input 3 [N, 100]
        masks_input, # Model input 4 (dict)
        original_lengths_input, # Model input 5 [N, 100]
        target_total_dg=target_total_dg_tensor_sim, # Target data 1 [N] (kcal/mol)
        original_window_dGs_list=original_window_dGs_list_sim, # Target data 2 (list of tensors, kcal/mol)
        original_dlambda_list=original_dlambda_list_sim, # Target data 3 (list of floats)
        reduction='mean' # Final loss reduction
    )


    # 6. 打印结果
    print("\nResults:")
    print(f"Input Data Shape: {input_data.shape}") # Should be [N, in_chans_simulated, 100, max_data]

    print(f"Predicted dGs per window Shape: {pred_dGs_per_window.shape}") # [N, 100]
    print(f"Target Total dG Shape (simulated): {target_total_dg_tensor_sim.shape}") # [N] (kcal/mol)
    print(f"Predicted Total dG Shape: {predicted_total_dg.shape}") # [N] (kcal/mol)
    print(f"Target Total dG Shape: {target_total_dg.shape}") # [N] (kcal/mol)
    print(f"Original Window dGs list length (simulated): {len(original_window_dGs_list_sim)}")
    for i, tg in enumerate(original_window_dGs_list_sim):
         print(f"  Sample {i} original window dGs shape: {tg.shape}") # [num_original_windows] (kcal/mol)
    print(f"Original dlambda list (simulated): {original_dlambda_list_sim}")


    print(f"\nTotal dG Loss (Mean over batch): {losses['total_dg_loss'].item():.4f}")
    print(f"Aggregated Window dG Loss (Mean over batch): {losses['agg_dg_loss'].item():.4f}")
    # Print the smoothness loss
    print(f"Smoothness Loss ({losses['smoothness_loss'].shape}): {losses['smoothness_loss'].item():.4f}")
    # Print the feature loss
    print(f"Feature Loss ({losses['feature_loss'].shape}): {losses['feature_loss'].item():.4f}")
    print(f"Combined Loss ({losses['combined_loss'].shape}): {losses['combined_loss'].item():.4f}") # Combined loss shape depends on reduction

    # You could print a sample of pred_dGs_per_window if desired
    b_pred_dg = np.random.randint(0, batch_size)
    print(f"\nRandom predicted dGs per window sample (batch={b_pred_dg}):")
    # Detach the tensor before converting to numpy
    print(pred_dGs_per_window[b_pred_dg, :10].detach().cpu().numpy()) # Print first 10 predicted dGs for a random sample

    # Print sample predicted and target total dG
    print(f"\nSample predicted total dG (kcal/mol): {predicted_total_dg.tolist()}")
    print(f"Sample target total dG (kcal/mol):   {target_total_dg.tolist()}")


    # 7. 检查GPU内存使用
    if device.type == 'cuda':
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**2
        max_allocated_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nGPU Memory Usage on {device}:")
        print(f"  Current memory allocated: {allocated_memory:.2f} MB")
        print(f"  Peak memory allocated: {max_allocated_memory:.2f} MB")

    # 8. 数据检查
    print("\n--- Data Inspection ---")
    if batch_size > 0:
        print("--- Raw Data (Sample 0 Overview) ---")
        print(f"Number of windows: {len(original_lambdas[0])}")

        # 随机选择5个窗口进行详细展示
        sample_indices = np.random.choice(len(original_lambdas[0]), size=min(5, len(original_lambdas[0])), replace=False)
        for idx in sorted(sample_indices):
            print(f"  Window {idx}: λ={original_lambdas[0][idx]:.2f}, Δλ={original_deltas[0][idx]:.2f}, Data points={original_data_lengths[0][idx]}")

        if len(raw_batch_data[0]) > 0:
            random_window = np.random.randint(0, len(raw_batch_data[0]))
            print(f"\nRandom raw window {random_window} tensor (shape {raw_batch_data[0][random_window].shape}):")
            # Print first 5 data points for the first few channels
            print(f"  Channels 0 to {min(3, raw_batch_data[0][random_window].shape[0]-1)} (first 5 points):\n{raw_batch_data[0][random_window][:min(4, raw_batch_data[0][random_window].shape[0]), :min(5, raw_batch_data[0][random_window].shape[1])].cpu().numpy()}")
            # Print last point for dG calculation check (only first 3 channels)
            if raw_batch_data[0][random_window].shape[1] > 0:
                print(f"  Last point (μ, σ, error): {raw_batch_data[0][random_window][:3, -1].cpu().numpy()}")

