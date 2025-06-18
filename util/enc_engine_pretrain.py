# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable, Optional, Dict, Any, List
from dataclasses import dataclass

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import pandas as pd
import os


@dataclass
class TrainingConfig:
    """训练配置类"""
    print_freq: int = 20
    default_save_path: str = '/nfs/export4_25T/ynlu/MAE-test/test_tool/output_dir/training_results.csv'
    log_frequency: int = 1  # 每多少个accum_iter记录一次日志


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.pred_total_dGs: List[float] = []
        self.true_total_dGs: List[float] = []
        self.system_labels: List[str] = []
    
    def update(self, pred_dg: torch.Tensor, true_dg: torch.Tensor, labels: List[str]):
        """更新预测结果"""
        self.pred_total_dGs.extend(pred_dg.detach().cpu().numpy())
        self.true_total_dGs.extend(true_dg.detach().cpu().numpy())
        self.system_labels.extend(labels)
    
    def get_results(self) -> Dict[str, List]:
        """获取收集的结果"""
        min_len = min(len(self.pred_total_dGs), len(self.true_total_dGs), len(self.system_labels))
        return {
            'predictions': self.pred_total_dGs[:min_len],
            'targets': self.true_total_dGs[:min_len],
            'labels': self.system_labels[:min_len]
        }


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, metric_logger, log_writer=None):
        self.metric_logger = metric_logger
        self.log_writer = log_writer
    
    def update_metrics(self, losses_dict: Dict[str, torch.Tensor], lr: float):
        """更新本地指标"""
        self.metric_logger.update(loss=losses_dict.get('combined_loss', 0.0).item() if isinstance(losses_dict.get('combined_loss'), torch.Tensor) else losses_dict.get('combined_loss', 0.0))
        self.metric_logger.update(total_dg_loss=losses_dict['total_dg_loss'].item())
        self.metric_logger.update(agg_dg_loss=losses_dict['agg_dg_loss'].item())
        self.metric_logger.update(smoothness_loss=losses_dict['smoothness_loss'].item())
        
        if 'feature_loss' in losses_dict:
            self.metric_logger.update(feature_loss=losses_dict['feature_loss'].item())
        
        self.metric_logger.update(lr=lr)
    
    def log_to_tensorboard(self, losses_dict: Dict[str, torch.Tensor], lr: float, step: int):
        """记录到TensorBoard"""
        if self.log_writer is None:
            return
        
        # 分布式同步
        loss_value = losses_dict.get('combined_loss', torch.tensor(0.0))
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        
        metrics_to_log = {
            'train_loss': misc.all_reduce_mean(loss_value),
            'train_total_dg_loss': misc.all_reduce_mean(losses_dict['total_dg_loss'].item()),
            'train_agg_dg_loss': misc.all_reduce_mean(losses_dict['agg_dg_loss'].item()),
            'train_smoothness_loss': misc.all_reduce_mean(losses_dict['smoothness_loss'].item()),
            'train_feature_loss': misc.all_reduce_mean(losses_dict.get('feature_loss', torch.tensor(0.0)).item()),
            'lr': lr
        }
        
        for metric_name, metric_value in metrics_to_log.items():
            self.log_writer.add_scalar(metric_name, metric_value, step)


def save_training_results(pred_dGs: List[float], true_dGs: List[float], labels: List[str], 
                         save_path: str = TrainingConfig.default_save_path) -> bool:
    """保存训练结果到CSV文件"""
    try:
        min_len = min(len(pred_dGs), len(true_dGs), len(labels))
        if min_len == 0:
            print("警告: 没有有效的训练结果数据")
            return False
        
        pred_dGs_safe = [float(p) for p in pred_dGs[:min_len]]
        true_dGs_safe = [float(t) for t in true_dGs[:min_len]]
        
        results_df = pd.DataFrame({
            'System': labels[:min_len],
            'Predicted_dG': pred_dGs_safe,
            'True_dG': true_dGs_safe,
            'Error': [abs(p - t) for p, t in zip(pred_dGs_safe, true_dGs_safe)]
        })
        
        results_df = results_df.sort_values(by='System').reset_index(drop=True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\n训练结果已保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"保存训练结果失败: {e}")
        return False


def print_training_summary(results: Dict[str, List]):
    """打印训练总结"""
    print("\n训练集总 dG 预测与真实值:")
    for pred, true, label in zip(results['predictions'], results['targets'], results['labels']):
        error = abs(pred - true)
        print(f"System {label}: Predicted_dG = {pred:.2f}, True_dG = {true:.2f}, Error = {error:.2f}")


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer: Optional[Any] = None,
                    args: Optional[Any] = None,
                    train_means: Optional[torch.Tensor] = None,
                    train_stds: Optional[torch.Tensor] = None,
                    config: Optional[TrainingConfig] = None
                    ) -> Dict[str, float]:
    """训练一个epoch"""
    if config is None:
        config = TrainingConfig()
    
    model.train(True)
    
    # 初始化指标记录器
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('agg_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('smoothness_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('feature_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    
    # 初始化组件
    training_logger = TrainingLogger(metric_logger, log_writer)
    metrics_collector = MetricsCollector()
    
    header = f'Epoch: [{epoch}]'
    accum_iter = getattr(args, 'accum_iter', 1)
    
    optimizer.zero_grad()
    
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')
    
    # 主训练循环
    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, config.print_freq, header)):
        (
            processed_data_dict, original_window_dGs_list, original_dlambda_list, 
            batched_target_total_dg, system_labels, batch_all_original_mu_targets, 
            batch_all_original_sigma_targets, batch_all_original_100_grid_indices
        ) = batch_data
        
        # 学习率调整
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # 数据转移到设备
        try:
            samples = processed_data_dict['data'].to(device, non_blocking=True)
            lambdas = processed_data_dict['lambdas'].to(device, non_blocking=True)
            deltas = processed_data_dict['deltas'].to(device, non_blocking=True)
            masks = {k: v.to(device, non_blocking=True) for k, v in processed_data_dict['masks'].items()}
            batched_target_total_dg = batched_target_total_dg.to(device, non_blocking=True)
            original_lengths = processed_data_dict['original_lengths'].to(device, non_blocking=True)
        except Exception as e:
            print(f"数据转移到设备失败: {e}")
            continue
        
        # 前向传播
        with torch.cuda.amp.autocast():
            losses_dict, pred_dGs_per_window, predicted_total_dg, target_total_dg = model(
                samples, lambdas, deltas, masks, original_lengths,
                target_total_dg=batched_target_total_dg,
                original_window_dGs_list=original_window_dGs_list,
                original_dlambda_list=original_dlambda_list,
                reduction='mean',
                all_original_mu_targets_list=batch_all_original_mu_targets,
                all_original_sigma_targets_list=batch_all_original_sigma_targets,
                all_original_100_grid_indices_list=batch_all_original_100_grid_indices
            )
            
            loss = losses_dict['combined_loss']
        
        # 收集预测结果
        metrics_collector.update(predicted_total_dg, target_total_dg, system_labels)
        
        # 损失检查
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        # 反向传播
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                   update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # 更新指标
        lr = optimizer.param_groups[0]["lr"]
        losses_dict_with_combined = dict(losses_dict)
        losses_dict_with_combined['combined_loss'] = loss_value
        training_logger.update_metrics(losses_dict_with_combined, lr)
        
        # TensorBoard日志记录
        if (data_iter_step + 1) % accum_iter == 0 and (data_iter_step + 1) % (accum_iter * config.log_frequency) == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            training_logger.log_to_tensorboard(losses_dict_with_combined, lr, epoch_1000x)
    
    # Epoch结束处理
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # 获取并展示结果
    training_results = metrics_collector.get_results()
    print_training_summary(training_results)
    
    # 保存结果
    save_training_results(
        training_results['predictions'], 
        training_results['targets'], 
        training_results['labels'],
        config.default_save_path
    )
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# 向后兼容性包装函数
def train_one_epoch_legacy(model: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, loss_scaler,
                          log_writer=None,
                          args=None,
                          train_means: torch.Tensor = None,
                          train_stds: torch.Tensor = None
                          ):
    """向后兼容的训练函数包装器"""
    return train_one_epoch(
        model, data_loader, optimizer, device, epoch, loss_scaler,
        log_writer, args, train_means, train_stds
    )