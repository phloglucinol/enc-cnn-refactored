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
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import pandas as pd
import os


def save_training_results(pred_dGs, true_dGs, labels, save_path):
    min_len = min(len(pred_dGs), len(true_dGs), len(labels))
    pred_dGs = [float(p) for p in pred_dGs[:min_len]]
    true_dGs = [float(t) for t in true_dGs[:min_len]]

    results_df = pd.DataFrame({
        'System': labels[:min_len],
        'Predicted_dG': pred_dGs,
        'True_dG': true_dGs,
        'Error': [abs(p - t) for p, t in zip(pred_dGs, true_dGs)]
    })

    results_df = results_df.sort_values(by='System').reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"\n训练结果已保存: {save_path}")


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    train_means: torch.Tensor = None,
                    train_stds: torch.Tensor = None
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('agg_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('smoothness_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('feature_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_total_dGs_list = []
    true_total_dGs_list = []
    all_system_labels = []

    for data_iter_step, (processed_data_dict, original_window_dGs_list, original_dlambda_list, batched_target_total_dg, system_labels, batch_all_original_mu_targets, batch_all_original_sigma_targets, batch_all_original_100_grid_indices) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = processed_data_dict['data'].to(device, non_blocking=True)
        lambdas = processed_data_dict['lambdas'].to(device, non_blocking=True)
        deltas = processed_data_dict['deltas'].to(device, non_blocking=True)
        masks = {k: v.to(device, non_blocking=True) for k, v in processed_data_dict['masks'].items()}
        batched_target_total_dg = batched_target_total_dg.to(device, non_blocking=True)
        original_lengths = processed_data_dict['original_lengths'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            losses_dict, pred_dGs_per_window, predicted_total_dg, target_total_dg = model(
                 samples,
                 lambdas,
                 deltas,
                 masks,
                 original_lengths,
                 target_total_dg=batched_target_total_dg,
                 original_window_dGs_list=original_window_dGs_list,
                 original_dlambda_list=original_dlambda_list,
                 reduction='mean',
                 all_original_mu_targets_list=batch_all_original_mu_targets,
                 all_original_sigma_targets_list=batch_all_original_sigma_targets,
                 all_original_100_grid_indices_list=batch_all_original_100_grid_indices
            )

            loss = losses_dict['combined_loss']

        pred_total_dGs_list.extend(predicted_total_dg.detach().cpu().numpy())
        true_total_dGs_list.extend(target_total_dg.detach().cpu().numpy())
        all_system_labels.extend(system_labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                    # clip_grad=args.max_grad_norm if hasattr(args, 'max_grad_norm') else 0.0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(total_dg_loss=losses_dict['total_dg_loss'].item())
        metric_logger.update(agg_dg_loss=losses_dict['agg_dg_loss'].item())
        metric_logger.update(smoothness_loss=losses_dict['smoothness_loss'].item())
        if 'feature_loss' in losses_dict:
             metric_logger.update(feature_loss=losses_dict['feature_loss'].item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        total_dg_loss_reduce = misc.all_reduce_mean(losses_dict['total_dg_loss'].item())
        agg_dg_loss_reduce = misc.all_reduce_mean(losses_dict['agg_dg_loss'].item())
        smoothness_loss_reduce = misc.all_reduce_mean(losses_dict['smoothness_loss'].item())
        feature_loss_reduce = misc.all_reduce_mean(losses_dict.get('feature_loss', torch.tensor(0.0)).item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_total_dg_loss', total_dg_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_agg_dg_loss', agg_dg_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_smoothness_loss', smoothness_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_feature_loss', feature_loss_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    print("\n训练集总 dG 预测与真实值:")
    min_len = min(len(pred_total_dGs_list), len(true_total_dGs_list), len(all_system_labels))
    for pred, true, label in zip(pred_total_dGs_list[:min_len], true_total_dGs_list[:min_len], all_system_labels[:min_len]):
        print(f"System {label}: Predicted_dG = {pred:.2f}, True_dG = {true:.2f}, Error = {abs(pred - true):.2f}")

    # training_save_path = f'/nfs/export4_25T/ynlu/MAE-test/test_tool/output_dir/training_results_epoch_{epoch}.csv'
    training_save_path = f'/nfs/export4_25T/ynlu/MAE-test/test_tool/output_dir/training_results.csv'
    save_training_results(pred_total_dGs_list, true_total_dGs_list, all_system_labels, training_save_path)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}