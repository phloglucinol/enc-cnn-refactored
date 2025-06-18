#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器管理模块

该模块定义了训练器类，负责：
1. 模型初始化和配置
2. 数据加载器创建
3. 优化器和学习率调度器设置
4. 训练循环管理
5. 模型保存和恢复

通过训练器类，将复杂的训练逻辑封装成清晰的接口。
"""

import os
import time
import json
import random
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import encoder_cnn_model
from util import test_lambda_emb_dataset
from util.test_lambda_emb_dataset import custom_collate_fn
from util.enc_engine_finetune import validate
from util.enc_engine_pretrain import train_one_epoch

from config import Config


class Trainer:
    """
    训练器类
    
    负责管理整个训练流程，包括模型初始化、数据加载、训练循环等。
    """
    
    def __init__(self, config: Config):
        """
        初始化训练器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        self.device = torch.device(config.training.device)
        
        # 初始化组件
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.loss_scaler = None
        self.data_loader_train = None
        self.data_loader_val = None
        self.log_writer = None
        
        # 训练统计
        self.best_val_loss = float('inf')
        self.train_means = None
        self.train_stds = None
        
        # 设置随机种子和环境
        self._setup_environment()
        
    def _setup_environment(self):
        """设置训练环境"""
        # 初始化分布式训练
        misc.init_distributed_mode(self.config.training)
        
        # 设置随机种子
        seed = self.config.training.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 启用CUDA优化
        cudnn.benchmark = True
        
        print(f'工作目录: {os.path.dirname(os.path.realpath(__file__))}')
        print(f'训练配置: {self.config}')
        
    def _create_datasets(self):
        """创建训练和验证数据集"""
        print("创建数据集...")
        
        # 创建训练数据集
        dataset_train = test_lambda_emb_dataset.CustomDataset(
            os.path.join(self.config.data.data_path, 'train'),
            subset_size=self.config.data.subset_size,
            processor_max_data=self.config.data.processor_max_data,
            processor_include_delta=self.config.data.processor_include_delta,
            num_random_subsets_per_system=self.config.data.num_random_subsets_per_system,
            per_lambda_max_points=self.config.data.per_lambda_max_points
        )
        
        # 创建验证数据集
        dataset_val = test_lambda_emb_dataset.CustomDataset(
            os.path.join(self.config.data.data_path, 'val'),
            subset_size=self.config.data.subset_size,
            processor_max_data=self.config.data.processor_max_data,
            processor_include_delta=self.config.data.processor_include_delta,
            num_random_subsets_per_system=self.config.data.num_random_subsets_per_system,
            per_lambda_max_points=self.config.data.per_lambda_max_points
        )
        
        print(f"训练数据集大小: {len(dataset_train)}")
        print(f"验证数据集大小: {len(dataset_val)}")
        
        return dataset_train, dataset_val
    
    def _create_data_loaders(self, dataset_train, dataset_val):
        """创建数据加载器"""
        print("创建数据加载器...")
        
        # 创建数据采样器
        if self.config.training.world_size > 1:  # 分布式训练
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:  # 单卡训练
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        # 创建训练数据加载器
        self.data_loader_train = DataLoader(
            dataset_train, 
            sampler=sampler_train,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        # 创建验证数据加载器  
        self.data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        print(f"训练批次数: {len(self.data_loader_train)}")
        print(f"验证批次数: {len(self.data_loader_val)}")
        
    def _calculate_dataset_statistics(self):
        """计算训练数据集的统计信息（均值和标准差）"""
        if not misc.is_main_process():
            # 非主进程使用默认值
            self.train_means = torch.zeros(3).to(self.device)
            self.train_stds = torch.ones(3).to(self.device)
            return
            
        print("\n计算训练数据统计信息...")
        
        # 用于累积统计的变量
        sum_of_features = torch.zeros(3).to(self.device)  # μ, σ², error通道的累积和
        sum_of_squares = torch.zeros(3).to(self.device)   # 平方和
        total_valid_points = torch.zeros(3).to(self.device)  # 有效点数
        
        # 创建临时数据加载器进行统计计算
        stats_loader = DataLoader(
            self.data_loader_train.dataset,
            sampler=self.data_loader_train.sampler,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(stats_loader):
                if batch_idx % 10 == 0:
                    print(f"处理统计批次 {batch_idx + 1}/{len(stats_loader)}")
                
                # 解析批次数据
                processed_data_dict = batch_data[0]
                data_tensor = processed_data_dict['data'].to(self.device)  # [N, C, 100, max_data]
                window_mask = processed_data_dict['masks']['window'].to(self.device)  # [N, 100]
                original_lengths = processed_data_dict['original_lengths'].to(self.device)  # [N, 100]
                
                # 检查数据通道数
                if data_tensor.shape[1] < 3:
                    print(f"警告: 数据张量只有 {data_tensor.shape[1]} 个通道，跳过统计计算")
                    self.train_means = torch.zeros(3).to(self.device)
                    self.train_stds = torch.ones(3).to(self.device)
                    return
                
                # 只考虑前3个通道 (μ, σ², error)
                data_for_stats = data_tensor[:, :3, :, :]  # [N, 3, 100, max_data]
                
                # 为每个通道计算有效数据掩码
                for c in range(3):
                    channel_data = data_for_stats[:, c, :, :]  # [N, 100, max_data]
                    
                    # 为每个样本和窗口创建数据点掩码
                    for i in range(channel_data.shape[0]):  # 遍历批次
                        for j in range(channel_data.shape[1]):  # 遍历窗口
                            if window_mask[i, j] > 0:  # 窗口有效
                                length = int(original_lengths[i, j].item())
                                if length > 0:
                                    # 获取有效数据点
                                    valid_data = channel_data[i, j, :length]
                                    
                                    # 累积统计
                                    sum_of_features[c] += valid_data.sum()
                                    sum_of_squares[c] += (valid_data ** 2).sum()
                                    total_valid_points[c] += valid_data.numel()
        
        # 防止除零错误
        total_valid_points[total_valid_points == 0] = 1
        
        # 计算均值和标准差
        self.train_means = sum_of_features / total_valid_points
        train_variances = (sum_of_squares / total_valid_points) - (self.train_means ** 2)
        train_variances[train_variances < 0] = 0  # 处理数值误差
        self.train_stds = torch.sqrt(train_variances)
        self.train_stds[self.train_stds == 0] = 1.0  # 防止除零
        
        print(f"计算得到的训练数据均值: {self.train_means.cpu().numpy()}")
        print(f"计算得到的训练数据标准差: {self.train_stds.cpu().numpy()}")
        
    def _create_model(self):
        """创建和初始化模型"""
        print("创建模型...")
        
        # 创建模型实例
        self.model = getattr(encoder_cnn_model, self.config.model.model_name)(
            norm_pix_loss=self.config.model.norm_pix_loss,
            train_means=self.train_means,
            train_stds=self.train_stds,
            in_chans=self.config.model.in_chans,
            total_dg_loss_weight=self.config.model.total_dg_loss_weight,
            agg_dg_loss_weight=self.config.model.agg_dg_loss_weight,
            smoothness_loss_weight=self.config.model.smoothness_loss_weight,
            feature_loss_weight=self.config.model.feature_loss_weight
        )
        
        self.model.to(self.device)
        
        # 处理分布式训练
        if self.config.training.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.config.training.local_rank], 
                find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model
        
        print(f"模型结构: {str(self.model_without_ddp)}")
        
        # 计算有效批次大小和学习率
        eff_batch_size = (self.config.training.batch_size * 
                         self.config.training.accum_iter * 
                         misc.get_world_size())
        
        print(f"基础学习率: {self.config.training.lr * 256 / eff_batch_size:.2e}")
        print(f"实际学习率: {self.config.training.lr:.2e}")
        print(f"有效批次大小: {eff_batch_size}")
        
    def _create_optimizer(self):
        """创建优化器和损失缩放器"""
        print("创建优化器...")
        
        # 创建参数组（支持权重衰减）
        param_groups = optim_factory.add_weight_decay(
            self.model_without_ddp, 
            self.config.training.weight_decay
        )
        
        # 创建AdamW优化器
        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.config.training.lr, 
            betas=(0.9, 0.95)
        )
        
        # 创建损失缩放器
        self.loss_scaler = NativeScaler()
        
        print(f"优化器: {type(self.optimizer).__name__}")
        print(f"学习率: {self.config.training.lr}")
        print(f"权重衰减: {self.config.training.weight_decay}")
        
    def _setup_logging(self):
        """设置日志记录"""
        if misc.is_main_process() and self.config.output.tensorboard_log:
            os.makedirs(self.config.output.log_dir, exist_ok=True)
            self.log_writer = SummaryWriter(log_dir=self.config.output.log_dir)
            print(f"TensorBoard日志将保存到: {self.config.output.log_dir}")
        else:
            self.log_writer = None
            
    def _load_checkpoint(self):
        """加载检查点"""
        if self.config.training.resume:
            print(f"从检查点恢复训练: {self.config.training.resume}")
            misc.load_model(
                args=self.config.training,
                model_without_ddp=self.model_without_ddp,
                optimizer=self.optimizer,
                loss_scaler=self.loss_scaler
            )
            
    def setup(self):
        """设置训练器的所有组件"""
        print("初始化训练器...")
        
        # 创建数据集和数据加载器
        dataset_train, dataset_val = self._create_datasets()
        self._create_data_loaders(dataset_train, dataset_val)
        
        # 计算数据集统计信息  
        self._calculate_dataset_statistics()
        
        # 创建模型
        self._create_model()
        
        # 创建优化器
        self._create_optimizer()
        
        # 设置日志
        self._setup_logging()
        
        # 加载检查点
        self._load_checkpoint()
        
        print("训练器初始化完成！")
        
    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        """训练一个轮次"""
        if self.config.training.world_size > 1:
            self.data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            self.model, 
            self.data_loader_train,
            self.optimizer, 
            self.device, 
            epoch, 
            self.loss_scaler,
            log_writer=self.log_writer,
            args=self.config.training,
            train_means=self.train_means,
            train_stds=self.train_stds
        )
        
        return train_stats
        
    def validate_one_epoch(self) -> Dict[str, Any]:
        """验证一个轮次"""
        val_stats = validate(
            self.data_loader_val,
            self.model,
            self.device,
            train_means=self.train_means,
            train_stds=self.train_stds
        )
        
        return val_stats
        
    def save_checkpoint(self, epoch: int, val_loss: float):
        """保存检查点"""
        if not misc.is_main_process():
            return
            
        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.config.output.output_dir:
                misc.save_model(
                    args=self.config.training,
                    model=self.model,
                    model_without_ddp=self.model_without_ddp,
                    optimizer=self.optimizer,
                    loss_scaler=self.loss_scaler,
                    epoch=epoch
                )
                print(f"保存最佳模型 - 轮次 {epoch}, 验证损失 {self.best_val_loss:.4f}")
                
    def log_metrics(self, epoch: int, train_stats: Dict, val_stats: Dict):
        """记录训练指标"""
        if not misc.is_main_process():
            return
            
        # 构建日志统计
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'best_val_loss': self.best_val_loss
        }
        
        # TensorBoard日志
        if self.log_writer is not None:
            self.log_writer.add_scalar('val/loss', val_stats['loss'], epoch)
            
            # 记录各种损失
            for loss_name in ['total_dg_loss', 'agg_dg_loss', 'smoothness_loss', 'feature_loss']:
                if loss_name in val_stats:
                    self.log_writer.add_scalar(f'val/{loss_name}', val_stats[loss_name], epoch)
                    
            self.log_writer.flush()
            
        # 文本日志
        if self.config.output.output_dir:
            log_file = os.path.join(self.config.output.output_dir, "training_log.txt")
            with open(log_file, mode="a", encoding="utf-8") as f:
                # 确保数值可以JSON序列化
                serializable_stats = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v) 
                    for k, v in log_stats.items()
                }
                f.write(json.dumps(serializable_stats) + "\n")
                
    def train(self):
        """执行完整的训练流程"""
        print(f"开始训练 {self.config.training.epochs} 个轮次")
        start_time = time.time()
        
        for epoch in range(self.config.training.start_epoch, self.config.training.epochs):
            print(f"\n{'='*50}")
            print(f"轮次 {epoch + 1}/{self.config.training.epochs}")
            print(f"{'='*50}")
            
            # 训练一个轮次
            train_stats = self.train_one_epoch(epoch)
            
            # 验证一个轮次
            val_stats = self.validate_one_epoch()
            
            # 保存检查点
            val_loss = val_stats['loss']
            self.save_checkpoint(epoch, val_loss)
            
            # 记录指标
            self.log_metrics(epoch, train_stats, val_stats)
            
            # 打印关键指标
            if misc.is_main_process():
                print(f"训练损失: {train_stats.get('loss', 'N/A'):.4f}")
                print(f"验证损失: {val_loss:.4f}")
                print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        # 训练完成
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'\n训练完成！总用时: {total_time_str}')
        
        # 关闭日志
        if self.log_writer is not None:
            self.log_writer.close()