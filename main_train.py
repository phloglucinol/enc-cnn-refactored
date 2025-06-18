#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子动力学自由能计算主训练脚本 (重构版)

这是重构后的主训练脚本，具有以下特点：
1. 代码结构清晰，职责分离明确
2. 使用配置管理系统统一管理参数
3. 通过训练器类封装复杂的训练逻辑
4. 完善的错误处理和日志记录
5. 良好的可扩展性和可维护性

该脚本基于Masked Autoencoder架构，结合CNN和Transformer来预测分子系统的自由能变化(ΔG)。

使用方法:
    python main_train.py --data_path /path/to/data --epochs 100 --batch_size 4

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
"""

import argparse
import json
import sys
from pathlib import Path

from config import Config, create_config_from_args
from trainer import Trainer


def get_args_parser():
    """
    构建命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser('分子动力学自由能计算训练脚本', add_help=True)
    
    # 数据相关参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--data_path', 
                           default='/nfs/export4_25T/ynlu/data/enc_cnn_dU_info_dataset/8-1-1/s0', 
                           type=str, help='数据集路径')
    data_group.add_argument('--subset_size', default=14, type=int,
                           help='数据子集大小（用于训练和验证）')
    data_group.add_argument('--num_random_subsets_per_system', default=20, type=int,
                           help='每个原始系统生成的随机子集数量')
    data_group.add_argument('--per_lambda_max_points', default=10, type=int,
                           help='每个lambda窗口采样的数据点数量（从开头开始）')
    data_group.add_argument('--processor_include_delta', action='store_true',
                           help='在处理后的数据中包含delta_lambda作为第4通道')
    data_group.add_argument('--num_workers', default=8, type=int,
                           help='数据加载器工作进程数')
    data_group.add_argument('--pin_mem', action='store_true',
                           help='在数据加载器中固定CPU内存，提高GPU传输效率')
    data_group.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True, processor_include_delta=True)
    
    # 模型相关参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--model', default='enc_cnn_chans3', type=str,
                            help='要训练的模型名称')
    model_group.add_argument('--input_size', default=(50, 100), type=tuple,
                            help='输入数据尺寸 (高度, 宽度)')
    model_group.add_argument('--in_chans', default=3, type=int,
                            help='输入通道数 (通常为3: μ, σ², error)')
    model_group.add_argument('--norm_pix_loss', action='store_true',
                            help='使用(逐补丁)归一化像素作为计算损失的目标')
    parser.set_defaults(norm_pix_loss=False)
    
    # 损失函数权重
    loss_group = parser.add_argument_group('损失函数权重')
    loss_group.add_argument('--total_dg_loss_weight', type=float, default=1.0,
                           help='总ΔG损失权重')
    loss_group.add_argument('--agg_dg_loss_weight', type=float, default=1.0,
                           help='聚合窗口ΔG损失权重')
    loss_group.add_argument('--feature_loss_weight', type=float, default=1.0,
                           help='特征损失权重（不同dlambda窗口μ/σ间的关系）')
    loss_group.add_argument('--smoothness_loss_weight', type=float, default=0.1,
                           help='平滑损失权重')
    
    # 训练相关参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--batch_size', default=4, type=int,
                            help='每个GPU的批次大小')
    train_group.add_argument('--epochs', default=100, type=int,
                            help='训练总轮数')
    train_group.add_argument('--accum_iter', default=1, type=int,
                            help='梯度累积迭代次数')
    
    # 优化器参数
    opt_group = parser.add_argument_group('优化器参数')
    opt_group.add_argument('--weight_decay', type=float, default=0.05,
                          help='权重衰减系数')
    opt_group.add_argument('--lr', type=float, default=None,
                          help='学习率（绝对值）')
    opt_group.add_argument('--blr', type=float, default=1e-3,
                          help='基础学习率: 绝对学习率 = 基础学习率 * 总批次大小 / 256')
    opt_group.add_argument('--min_lr', type=float, default=0.0,
                          help='循环调度器的最小学习率')
    opt_group.add_argument('--warmup_epochs', type=int, default=40,
                          help='学习率预热轮数')
    
    # 系统参数
    sys_group = parser.add_argument_group('系统参数')
    sys_group.add_argument('--device', default='cuda', help='训练/测试设备')
    sys_group.add_argument('--seed', default=0, type=int, help='随机种子')
    sys_group.add_argument('--resume', default='', help='从检查点恢复训练')
    sys_group.add_argument('--start_epoch', default=0, type=int, help='开始轮数')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output_dir', default='./output_dir',
                             help='模型保存路径，为空则不保存')
    output_group.add_argument('--log_dir', default='./output_dir',
                             help='TensorBoard日志路径')
    
    # 分布式训练参数
    dist_group = parser.add_argument_group('分布式训练参数')
    dist_group.add_argument('--world_size', default=1, type=int,
                           help='分布式进程数')
    dist_group.add_argument('--local_rank', default=-1, type=int,
                           help='本地进程rank')
    dist_group.add_argument('--dist_on_itp', action='store_true',
                           help='在ITP上进行分布式训练')
    dist_group.add_argument('--dist_url', default='env://',
                           help='分布式训练设置URL')
    
    return parser


def main():
    """主函数：解析参数、创建配置、启动训练"""
    try:
        # 解析命令行参数
        parser = get_args_parser()
        args = parser.parse_args()
        
        # 创建配置对象
        config = create_config_from_args(args)
        
        # 创建输出目录并保存配置
        if config.output.output_dir:
            Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
            config_file = Path(config.output.output_dir) / 'config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                # 将配置对象转换为字典进行保存
                config_dict = {
                    'data': config.data.__dict__,
                    'model': config.model.__dict__,
                    'training': config.training.__dict__,
                    'output': config.output.__dict__
                }
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            print(f"配置已保存到: {config_file}")
        
        # 打印配置信息
        config.print_config()
        
        # 创建训练器
        trainer = Trainer(config)
        
        # 设置训练器组件
        trainer.setup()
        
        # 开始训练
        trainer.train()
        
        print("训练成功完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()