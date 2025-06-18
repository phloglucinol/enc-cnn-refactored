#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置编码工具函数

该模块提供了2D正弦-余弦位置编码的实现，用于Vision Transformer模型。

基于Meta的MAE项目实现。
"""

import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    生成2D正弦-余弦位置编码
    
    Args:
        embed_dim: 嵌入维度
        grid_size: 网格大小（假设为正方形网格）
        cls_token: 是否包含类别token的位置编码
    
    Returns:
        pos_embed: 位置编码数组，形状为 [grid_size*grid_size, embed_dim] 或 [1+grid_size*grid_size, embed_dim]
    """
    grid_h = grid_size
    grid_w = grid_size
    
    grid_h_coords = np.arange(grid_h, dtype=np.float32)
    grid_w_coords = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_coords, grid_h_coords)  # 这里的w和h顺序很重要
    grid = np.stack(grid, axis=0)  # [2, grid_h, grid_w]
    
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从网格坐标生成2D位置编码
    
    Args:
        embed_dim: 嵌入维度
        grid: 网格坐标，形状为 [2, 1, grid_h, grid_w]
    
    Returns:
        pos_embed: 位置编码数组，形状为 [grid_h*grid_w, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    # 使用一半的维度编码grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    # 使用另一半的维度编码grid_w  
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # [H*W, D]
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从1D位置生成正弦-余弦位置编码
    
    Args:
        embed_dim: 嵌入维度
        pos: 位置数组，形状为 [M,]
    
    Returns:
        emb: 位置编码数组，形状为 [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2]
    
    pos = pos.reshape(-1)  # [M]
    out = np.einsum('m,d->md', pos, omega)  # [M, D/2], 外积
    
    emb_sin = np.sin(out)  # [M, D/2]
    emb_cos = np.cos(out)  # [M, D/2]
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M, D]
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    生成1D正弦-余弦位置编码
    
    Args:
        embed_dim: 嵌入维度
        length: 序列长度
        cls_token: 是否包含类别token的位置编码
    
    Returns:
        pos_embed: 位置编码数组，形状为 [length, embed_dim] 或 [1+length, embed_dim]
    """
    pos = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed