#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块

该模块包含了项目中使用的所有深度学习模型定义。

主要模型：
- encoder_cnn_model: 基于Transformer和CNN的分子自由能预测模型

使用方法：
    from models import enc_cnn_chans3, enc_cnn_chans4
    
    # 创建3通道模型
    model3 = enc_cnn_chans3(
        train_means=train_means,
        train_stds=train_stds,
        total_dg_loss_weight=1.0
    )
    
    # 创建4通道模型  
    model4 = enc_cnn_chans4(
        train_means=train_means,
        train_stds=train_stds,
        total_dg_loss_weight=1.0
    )
"""

from .encoder_cnn_model import (
    enc_cnn_chans3,
    enc_cnn_chans4,
    MaskedAutoencoderViT,
    AdaptiveLambdaEncoding,
    EmbeddingStrategy3Chans,
    EmbeddingStrategy4Chans,
    BaseEmbeddingModule
)

__all__ = [
    'enc_cnn_chans3',
    'enc_cnn_chans4', 
    'MaskedAutoencoderViT',
    'AdaptiveLambdaEncoding',
    'EmbeddingStrategy3Chans',
    'EmbeddingStrategy4Chans',
    'BaseEmbeddingModule'
]