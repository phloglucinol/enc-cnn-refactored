#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子动力学自由能计算主训练脚本

该脚本基于Masked Autoencoder架构，结合CNN和Transformer来预测分子系统的自由能变化(ΔG)。
主要功能包括：
1. 加载和预处理分子动力学数据
2. 训练深度学习模型
3. 验证模型性能并保存最佳模型

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
--------------------------------------------------------
References:
DeiT: https://github.com/facebookresearch/deit
BEiT: https://github.com/microsoft/unilm/tree/master/beit
--------------------------------------------------------
"""
# 系统和工具库导入
import argparse  # 命令行参数解析
import datetime  # 日期时间处理
import json      # JSON数据处理
import numpy as np  # 数值计算
import os        # 操作系统接口
import time      # 时间处理
from pathlib import Path  # 路径操作
import random    # 随机数生成

# PyTorch深度学习框架
import torch
import torch.backends.cudnn as cudnn  # CUDA深度神经网络库
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录
# 数据集相关导入
from util import test_dataset as test_dataset  # 数据集处理模块
from util import test_dataset_split as test_dataset_split  # 数据集分割模块
from torch.utils.data import DataLoader  # PyTorch数据加载器

# timm库（PyTorch Image Models）用于预训练模型和优化器
import timm
assert timm.__version__ == "0.3.2"  # 版本检查确保兼容性
import timm.optim.optim_factory as optim_factory  # 优化器工厂

# 项目工具模块
import util.misc as misc  # 杂项工具函数
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # 梯度缩放器

# 模型定义模块
import models_mae_test_v2     # MAE模型版本2
import models_mae_test_v3     # MAE模型版本3
import encoder_cnn_model_test_v1  # 编码器CNN模型版本1

# 导入我们修改后的数据集文件和其配套的 collate_fn
from util import test_lambda_emb_dataset
from util.test_lambda_emb_dataset import custom_collate_fn

# from util.engine_finetune_test_origin import validate

# from util.engine_finetune_test import validate
from util.enc_engine_finetune import validate
from util.enc_engine_pretrain import train_one_epoch


def get_args_parser():
    """
    构建命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser('MAE training', add_help=False)
    # 训练超参数设置
    parser.add_argument('--batch_size', default=4, type=int,
                        help='每个GPU的批次大小 (有效批次大小 = batch_size * accum_iter * GPU数量)')  # 默认批次大小
    parser.add_argument('--epochs', default=100, type=int,
                        help='训练总轮数')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积迭代次数（用于在内存限制下增加有效批次大小）')

    # 模型参数配置
    parser.add_argument('--model', default='enc_cnn_chans3', type=str, metavar='MODEL',
                        help='要训练的模型名称')

    parser.add_argument('--input_size', default=(50, 100), type=int,
                        help='输入数据尺寸 (高度, 宽度)')
    
    parser.add_argument('--in_chans', default=3, type=int,
                        help='输入通道数 (通常为3: μ, σ², error)')

    # Add argument for the number of random subsets per system
    parser.add_argument('--num_random_subsets_per_system', type=int, default=20,
                        help='Number of random subsets to generate per original system (complex/ligand).')

    parser.add_argument('--subset_size', default=14, type=int,
                        help='size of data subset (for training and validation)')

    parser.add_argument('--per_lambda_max_points', default=10, type=int,
                        help='Number of data points to sample per lambda window (from the beginning). If None, use processor_max_data.')
    
    parser.add_argument('--total_dg_loss_weight', type=float, default=0,
                        help='Weight for total dG loss')

    parser.add_argument('--agg_dg_loss_weight', type=float, default=0,
                        help='Weight for aggregated window dG loss')

    parser.add_argument('--feature_loss_weight', type=float, default=1,
                        help='Weight for feature loss (relationship between mu/sigma of different dlambda windows)')

    parser.add_argument('--smoothness_loss_weight', type=float, default=0,
                        help='Weight for smoothness loss')

    parser.add_argument('--processor_include_delta', action='store_true',
                        help='Include delta_lambda as a 4th channel in processed data.')
    parser.set_defaults(processor_include_delta=True)

    # parser.add_argument('--mask_ratio', default=0.98, type=float,
    #                     help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/nfs/export4_25T/ynlu/data/enc_cnn_dU_info_dataset/8-1-1/s0', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)  # default=10
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # 分布式训练初始化
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) # Set seed for Python's random module as well for reproducibility of random sampling

    cudnn.benchmark = True

    # 加载训练集和验证集
    # Use the modified CustomDataset
    # Pass the new argument to the CustomDataset constructor
    dataset_train = test_lambda_emb_dataset.CustomDataset(os.path.join(args.data_path, 'train'),
                                                          subset_size=args.subset_size,
                                                          processor_max_data=50, # Assuming max_data needed
                                                          processor_include_delta=args.processor_include_delta,
                                                          num_random_subsets_per_system=args.num_random_subsets_per_system,
                                                          per_lambda_max_points=args.per_lambda_max_points
                                                         )

    # Pass the new argument to the CustomDataset constructor for validation set as well
    dataset_val = test_lambda_emb_dataset.CustomDataset(os.path.join(args.data_path, 'val'),
                                                        subset_size=args.subset_size,
                                                        processor_max_data=50, # Assuming max_data needed
                                                        processor_include_delta=args.processor_include_delta,
                                                        num_random_subsets_per_system=args.num_random_subsets_per_system,
                                                        per_lambda_max_points=args.per_lambda_max_points
                                                       )

    # dataset_train = test_dataset_split.CustomDataset(os.path.join(args.data_path, 'train'))
    # dataset_val = test_dataset_split.CustomDataset(os.path.join(args.data_path, 'val'))

    # mean_train = dataset_train.mean
    # std_train = dataset_train.std
    # print('mean train: {}'.format(mean_train))
    # print('std train: {}'.format(std_train))

    # mean_val = dataset_val.mean
    # std_val = dataset_val.std
    # print('mean val: {}'.format(mean_val))
    # print('std val: {}'.format(std_val))
    
    # Note: mean/std calculation needs to be adjusted if you were using these from the old datasets.
    # The current CustomDataset doesn't calculate mean/std of the processed data automatically.
    # If standardization is needed based on the *processed* data distribution, you'd need to implement that
    # after processing or calculate it separately over the full dataset.
    # For now, we will assume standardization might be handled differently or not used in the new model.
    # Removing old mean/std print statements as they might be misleading.
    # print('mean train: {}'.format(mean_train))
    # print('std train: {}'.format(std_train))
    # print('mean val: {}'.format(mean_val))
    # print('std val: {}'.format(std_val))

    # 数据采样器
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()  # world_size:GPU数量
        global_rank = misc.get_rank()  # rank:当前在第几张卡上
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_val = %s" % str(sampler_val)) # Add print for validation sampler
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 数据加载器 (用于训练和验证循环)
    # Use the custom_collate_fn here
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # --- 计算训练集数据的均值和标准差 (仅针对 μ, σ², error 这3个通道的有效数据点) ---
    # 创建一个临时的 DataLoader，仅用于统计计算
    # 注意：如果分布式训练，每个进程只计算其自己的数据子集的统计量，这可能不是我们想要的。
    # 正确的做法是在所有进程中收集数据点，然后计算全局统计量。或者，假设数据分布在不同进程间是相似的，
    # 简单起见，这里先只在主进程（或其他某个进程）计算，然后广播（如果需要）。
    # 对于非分布式，直接遍历 data_loader_train 即可。
    
    # TODO: For true distributed training, need to aggregate counts and sums across all ranks.
    # For simplicity in this edit, we assume non-distributed or calculate on rank 0 and use it everywhere.
    
    if misc.is_main_process():
        print("\nCalculating training data statistics...")
        sum_of_features = torch.zeros(3).to(device) # Sum for μ, σ², error channels
        sum_of_squares = torch.zeros(3).to(device) # Sum of squares for these channels
        total_valid_points = torch.zeros(3).to(device) # Count of valid points for each channel

        # Create a temporary data loader to iterate through the dataset
        # Use a large batch size for potentially faster calculation
        stats_data_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, # Use the same sampler to respect distribution if needed
            batch_size=args.batch_size * args.accum_iter, # Use effective batch size for stats? Or just a large batch? Let's use a large one.
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False, # Include all data
            collate_fn=custom_collate_fn
        )
        
        # Disable gradient calculation for efficiency
        with torch.no_grad():
            for data_iter_step, batch_data in enumerate(stats_data_loader):
                # Assuming processed_data_dict is the first element in the batch_data tuple/list
                processed_data_dict = batch_data[0]
                # We ignore the rest of the elements in batch_data as they are not needed for stats calculation.

                # processed_data_dict contains 'data' [N, C, 100, max_data], 'masks'['window'] [N, 100], 'original_lengths' [N, 100]
                data_tensor = processed_data_dict['data'].to(device) # [N, C, 100, max_data]
                window_mask = processed_data_dict['masks']['window'].to(device) # [N, 100]
                original_lengths = processed_data_dict['original_lengths'].to(device) # [N, 100]

                # Ensure data tensor has at least 3 channels for stats calculation
                if data_tensor.shape[1] < 3:
                     print(f"Warning: Data tensor has only {data_tensor.shape[1]} channels, expected at least 3 for stats calculation. Skipping stats.")
                     # Fallback to default stats or handle error
                     train_means = torch.zeros(3).to(device)
                     train_stds = torch.ones(3).to(device)
                     break # Exit stats calculation loop
                
                # Only consider the first 3 channels (μ, σ², error)
                data_for_stats = data_tensor[:, :3, :, :] # [N, 3, 100, max_data]
                
                # Expand mask and lengths for broadcasting
                # Mask is [N, 100], need [N, 1, 100, 1] for data [N, 3, 100, max_data]
                window_mask_expanded = window_mask.unsqueeze(1).unsqueeze(-1) # [N, 1, 100, 1]
                
                # Lengths are [N, 100], need to create a mask for the last dimension
                # Resulting mask should be [N, 1, 100, max_data]
                points_mask = torch.zeros_like(data_for_stats, dtype=torch.bool, device=device) # [N, 3, 100, max_data]
                for i in range(data_for_stats.shape[0]): # Iterate over batch size
                     for j in range(data_for_stats.shape[2]): # Iterate over windows (100)
                          length = int(original_lengths[i, j].item())
                          # Apply point mask based on original length, and window mask
                          if window_mask[i, j] > 0 and length > 0:
                                points_mask[i, :, j, :length] = True

                # Combine window mask and points mask
                # window_mask_expanded is [N, 1, 100, 1], points_mask is [N, 3, 100, max_data]
                # Broadcasing will handle the dimensions
                # Convert window_mask_expanded to boolean before bitwise AND
                effective_mask = window_mask_expanded.bool() & points_mask # [N, 3, 100, max_data]


                # Apply mask and flatten the valid data points for each channel
                masked_data = data_for_stats[effective_mask] # This flattens all valid points across batch, windows, and max_data.
                                                               # The shape is [num_total_valid_points, 3] if indexed correctly.
                                                               # Let's apply mask channel-wise for clarity and to get channel-specific counts/sums.
                
                for c in range(3): # Iterate over channels (μ, σ, error)
                     channel_data = data_for_stats[:, c, :, :] # [N, 100, max_data]
                     channel_mask = effective_mask[:, c, :, :] # [N, 100, max_data]
                     
                     valid_channel_data = channel_data[channel_mask] # Flattened tensor of valid data points for this channel
                     
                     sum_of_features[c] += valid_channel_data.sum()
                     sum_of_squares[c] += (valid_channel_data ** 2).sum()
                     total_valid_points[c] += valid_channel_data.numel() # Count the number of elements

            # Handle potential division by zero if a channel has no valid points
            total_valid_points[total_valid_points == 0] = 1 # Prevent division by zero, results in 0 mean and std for that channel
            
            train_means = sum_of_features / total_valid_points
            # Variance = (Sum of Squares / N) - Mean^2
            train_variances = (sum_of_squares / total_valid_points) - (train_means ** 2)
            # Handle potential negative variance due to floating point errors
            train_variances[train_variances < 0] = 0
            train_stds = torch.sqrt(train_variances)
            
            # Handle channels with zero variance (all values were the same). Set std to 1 to prevent division by zero during normalization.
            train_stds[train_stds == 0] = 1.0
            
            print(f"Calculated train_means: {train_means.cpu().numpy()}")
            print(f"Calculated train_stds:  {train_stds.cpu().numpy()}")
    else:
         # Non-main processes will have default mean 0 and std 1 or receive broadcasted values.
         # For now, let's initialize them to 0/1 on other ranks as a placeholder.
         # TODO: Implement broadcasting if necessary for true distributed training.
         train_means = torch.zeros(3).to(device)
         train_stds = torch.ones(3).to(device)
         print(f"Rank {misc.get_rank()}: Initialized train_means/stds as placeholders (assuming broadcast or rank 0 calc).")


    # 定义模型

    # model = encoder_cnn_model_test_v1.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    # # Pass train_means and train_stds to the model constructor
    # model = encoder_cnn_model_test_v1.__dict__[args.model](
    #     img_size=(100, 50), # Assuming this is needed by the model constructor
    #     in_chans=3 if not args.processor_include_delta else 4, # Pass correct input channels
    #     norm_pix_loss=args.norm_pix_loss,
    #     target_output_windows=100, # Explicitly pass target window count
    #     total_dg_loss_weight=getattr(args, 'total_dg_loss_weight', 1.0), # Allow overriding via args
    #     agg_dg_loss_weight=getattr(args, 'agg_dg_loss_weight', 1.0), # Allow overriding via args
    #     train_means=train_means, # Pass calculated means
    #     train_stds=train_stds # Pass calculated stds
    # )

    model = encoder_cnn_model_test_v1.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        train_means=train_means, # Pass calculated means
        train_stds=train_stds, # Pass calculated stds
        in_chans=args.in_chans,
        total_dg_loss_weight=args.total_dg_loss_weight, # Pass from args
        agg_dg_loss_weight=args.agg_dg_loss_weight, # Pass from args
        smoothness_loss_weight=args.smoothness_loss_weight, # Pass from args
        feature_loss_weight=args.feature_loss_weight # Pass new argument from args
    )
    model.to(device)

    # 假设模型没有使用分布式数据并行（Distributed Data Parallel, DDP），因此直接赋值
    # IMPORTANT: If using DDP, make sure model_without_ddp is the .module attribute
    model_without_ddp = model
    # model_without_ddp = model.module # Use this if model is wrapped in DDP immediately
    print("Model = %s" % str(model_without_ddp))

    """
    misc.get_world_size(): 获取分布式训练中的进程数（即 GPU 的数量）
    """
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    # 分布数据并行
    if args.distributed:
        # Note: If calculating stats only on rank 0, need to broadcast them before DDP setup
        # or ensure the model gets the correct stats instance on all ranks.
        # A simple way is to put the stats calculation *before* DDP and ensure train_means/stds are on device.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # Update model_without_ddp to refer to the unwrapped model
        model_without_ddp = model.module

    # 优化器设置
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # 加载预训练模型（如果有）
    # Note: If loading a checkpoint from a model without these stats attributes,
    # the load_state_dict might warn about missing keys. This is usually fine
    # as these are just parameters used for normalization, not learnable weights.
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # 初始化最佳验证损失
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 训练一个 epoch
        # Modify train_one_epoch to accept the new data format
        # train_stats = train_one_epoch(
        #     model, data_loader_train,
        #     optimizer, device, epoch, loss_scaler, log_writer=log_writer,
        #     args=args, mean=mean_train, std=std_train) # Remove mean/std if not used in train_one_epoch

        
        # Call train_one_epoch with the updated data_loader output
        # The train_one_epoch function itself needs to be modified to unpack the data correctly
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, log_writer=log_writer,
            args=args,
            train_means=train_means, # Pass train means
            train_stds=train_stds # Pass train stds
        )  # Pass train_means and train_stds to train_one_epoch

        # 验证模型
        # val_stats = validate(data_loader_val, model, device)
        # val_stats = validate(data_loader_val, model, device, mean=mean_val, std=std_val) # Remove mean/std if not used

        # Call validate with the updated data_loader output
        # The validate function itself needs to be modified to unpack the data correctly
        # Pass train_means and train_stds to validate
        val_stats = validate(data_loader_val, model, device,
                             train_means=train_means, # Pass train means
                             train_stds=train_stds # Pass train stds
        )

        val_loss = val_stats['loss']  # 从 validate 返回的字典中获取 'val_loss'

        # 保存最佳模型
        # Only save checkpoint on main process
        if misc.is_main_process():
            # Correct condition for saving: Check if current val_loss is better than best_val_loss
            if val_loss < best_val_loss:  # 注意这里是小于号,因为我们希望损失越小越好
                best_val_loss = val_loss
                if args.output_dir:
                    # Include epoch in save path or filename for clarity, or use a fixed 'best' name
                    # Let's save the model checkpoint explicitly with epoch number for tracking
                    # Example: epoch_001.pth, epoch_005.pth etc.
                    # Or save the best model with a specific name like 'checkpoint-best.pth'
                    # The misc.save_model function already handles naming with epoch.
                    # Let's ensure it saves when validation loss improves.
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                    )
                    print(f"Saving best model at epoch {epoch} with validation loss {best_val_loss:.4f}")
            # Also save checkpoint periodically, regardless of performance (e.g., every few epochs)
            # Add a condition for periodic saving if needed
            # Example: if epoch % args.save_freq == 0: misc.save_model(...)


        # 日志记录 (仅在主进程进行)
        if misc.is_main_process():
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'best_val_loss': best_val_loss # Log best validation loss
            }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.add_scalar('val/loss', val_loss, epoch)  # 记录验证损失而不是准确率
                    # Log validation losses individually
                    if 'total_dg_loss' in val_stats:
                        log_writer.add_scalar('val/total_dg_loss', val_stats['total_dg_loss'], epoch)
                    if 'agg_dg_loss' in val_stats:
                        log_writer.add_scalar('val/agg_dg_loss', val_stats['agg_dg_loss'], epoch)
                    if 'smoothness_loss' in val_stats:
                         log_writer.add_scalar('val/smoothness_loss', val_stats['smoothness_loss'], epoch)
                    if 'feature_loss' in val_stats: # Log the new feature loss
                         log_writer.add_scalar('val/feature_loss', val_stats['feature_loss'], epoch)
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    # Convert numpy floats in log_stats to standard Python floats for JSON serialization
                    log_stats_serializable = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in log_stats.items()}
                    f.write(json.dumps(log_stats_serializable) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # Save arguments to a file for reference
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    main(args)
