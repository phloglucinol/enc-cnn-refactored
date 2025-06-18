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
from typing import Iterable, Optional

import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

import pandas as pd
import os


def train_one_epoch(model: torch.nn.Module,  # criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, val_data_loader: Optional[Iterable] = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10  # 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
            # samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # outputs = model(samples)
            # loss = criterion(outputs, targets)

            loss, output, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 在每个训练 epoch 结束后进行验证
    # NOTE: This calls the OLD validate function without the new data format.
    # We should remove this call from train_one_epoch and let main.py handle validation calls.
    # if val_data_loader is not None:
    #     validate(val_data_loader, model, device) # <- REMOVE THIS LINE

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def validate(data_loader, model, device, mean, std):
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Validation:'
#
#     # 初始化总损失
#     total_loss = 0.0
#     num_samples = 0
#
#     model.eval()
#
#     # for batch in metric_logger.log_every(data_loader, 10, header):
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch[0]
#         labels =batch[1]
#         print(f'system_labels={labels}')
#
#         images = images.to(device, non_blocking=True)
#         # target = target.to(device, non_blocking=True)
#
#         with torch.cuda.amp.autocast():
#             # output = model(images)
#             loss, dG_loss, system_dg = model(images, label=labels, mean=mean, std=std)
#             print('pred_dG:', system_dg.detach().cpu().numpy())  # detach()可以分离计算图，更节省内存
#
#         # 累加损失和样本数
#         total_loss += loss.item() * images.size(0)
#         num_samples += images.size(0)
#
#         # batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#
#     # 计算平均损失
#     avg_loss = total_loss / num_samples
#
#     metric_logger.synchronize_between_processes()
#     print('Validation loss: {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
#
#     return {'loss': avg_loss}


# Add a new helper function to save per-window dG results
def save_per_window_dG_results_for_plot(system_labels, aligned_lambdas_list, pred_dGs_list, true_dGs_list, save_path):
    """
    Saves per-window dG predictions and true values for plotting.

    Args:
        system_labels (list): List of system labels (strings).
        aligned_lambdas_list (list): List of numpy arrays, each [100] representing aligned lambda values.
        pred_dGs_list (list): List of numpy arrays, each [100] representing predicted dGs per window.
        true_dGs_list (list): List of numpy arrays, each [100] representing true dGs per window (aligned, with NaN).
        save_path (str): Path to save the CSV file.
    """
    records = []
    for i in range(len(system_labels)):
        system_label = system_labels[i]
        lambdas = aligned_lambdas_list[i]
        pred_dGs = pred_dGs_list[i]
        true_dGs = true_dGs_list[i]

        for j in range(len(lambdas)):
            records.append({
                'System': system_label,
                'Lambda': lambdas[j],
                'Predicted_dG_Window': pred_dGs[j],
                'True_dG_Window': true_dGs[j]
            })

    if not records:
        print("No per-window dG data to save.")
        return

    results_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"\n每个窗口的 dG 结果已保存用于绘图: {save_path}")


# --------------------------------------
@torch.no_grad()
def validate(data_loader, model, device,
             train_means: torch.Tensor = None,
             train_stds: torch.Tensor = None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    # Use the new loss metrics from the model's return dict
    metric_logger.add_meter('total_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('agg_dg_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('smoothness_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('feature_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_abs_error', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))

    header = 'Validation:'

    # 初始化列表来存储预测值和真实值
    # Store predicted total dG and target total dG
    pred_total_dGs_list = []
    true_total_dGs_list = []
    all_system_labels = []  # 添加这行来初始化列表

    # New lists to collect per-window data for plotting
    all_systems_pred_dGs_per_window = []
    all_systems_true_dGs_per_window_aligned = []
    all_systems_aligned_lambdas = []
    all_systems_labels_for_windows = []

    model.eval()

    # Unpack the data from the collate_fn output, including the lists for all original windows
    for data_iter_step, (processed_data_dict, original_window_dGs_list, original_dlambda_list, batched_target_total_dg, system_labels, batch_all_original_mu_targets, batch_all_original_sigma_targets, batch_all_original_100_grid_indices) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        # Extract necessary tensors from processed_data_dict and move to device
        samples = processed_data_dict['data'].to(device, non_blocking=True) # [N, C, 100, max_data]
        lambdas = processed_data_dict['lambdas'].to(device, non_blocking=True) # [N, 100]
        deltas = processed_data_dict['deltas'].to(device, non_blocking=True) # [N, 100]
        masks = {k: v.to(device, non_blocking=True) for k, v in processed_data_dict['masks'].items()} # Dict of [N, 100] masks
        original_lengths = processed_data_dict['original_lengths'].to(device, non_blocking=True) # [N, 100]

        # Move target tensor to device
        batched_target_total_dg = batched_target_total_dg.to(device, non_blocking=True)

        # Note: The lists (original_window_dGs_list, original_dlambda_list,
        # batch_all_original_mu_targets, batch_all_original_sigma_targets, batch_all_original_100_grid_indices)
        # are lists of tensors and should NOT be moved to device here.
        # They will be moved individually inside calculate_loss if needed.

        with torch.cuda.amp.autocast():
            # Call model - it now returns the losses dictionary, pred_dGs_per_window, predicted_total_dg, and target_total_dg
            # Pass original_lengths to the model's forward method
            # Pass the lists for all original window targets/indices to the model's forward method
            losses_dict, pred_dGs_per_window, predicted_total_dg, target_total_dg = model(
                  samples, # [N, C, 100, max_data]
                  lambdas, # [N, 100]
                  deltas, # [N, 100]
                  masks, # Dict of masks
                  original_lengths, # [N, 100] - Pass original_lengths
                  target_total_dg=batched_target_total_dg, # Target data 1 [N]
                  original_window_dGs_list=original_window_dGs_list, # Target data 2 (list)
                  original_dlambda_list=original_dlambda_list, # Target data 3 (list)
                  reduction='mean', # Use mean reduction for batch loss
                  all_original_mu_targets_list=batch_all_original_mu_targets, # Pass the list
                  all_original_sigma_targets_list=batch_all_original_sigma_targets, # Pass the list
                  all_original_100_grid_indices_list=batch_all_original_100_grid_indices # Pass the list
             )

        # Update metrics using the dict returned by model
        # The losses_dict values already account for batch size (mean reduction)
        metric_logger.update(loss=losses_dict['combined_loss'].item())
        metric_logger.update(total_dg_loss=losses_dict['total_dg_loss'].item())
        metric_logger.update(agg_dg_loss=losses_dict['agg_dg_loss'].item())
        metric_logger.update(smoothness_loss=losses_dict['smoothness_loss'].item())
        # Update metric for feature loss
        if 'feature_loss' in losses_dict: # Check if key exists
             metric_logger.update(feature_loss=losses_dict['feature_loss'].item())
        
        # Calculate Mean Absolute Error (MAE) for total dG prediction
        abs_error_per_sample = torch.abs(predicted_total_dg - target_total_dg) # [N]
        mean_abs_error_batch = torch.mean(abs_error_per_sample) # Scalar for this batch
        metric_logger.update(mean_abs_error=mean_abs_error_batch.item()) # Update smoothed value
        

        # Store predicted and true total dGs for comparison
        pred_total_dGs_list.extend(predicted_total_dg.cpu().numpy())
        true_total_dGs_list.extend(target_total_dg.cpu().numpy())

        # Store labels
        all_system_labels.extend(system_labels)

        # --- Collect per-window dG data for plotting ---
        # pred_dGs_per_window has shape [N, 100, 1]. Squeeze to [N, 100]
        pred_dGs_per_window_squeezed = pred_dGs_per_window.squeeze(-1) # [N, 100]

        for i in range(len(system_labels)): # Iterate through each system in the batch
            current_system_label = system_labels[i]
            aligned_lambdas = lambdas[i].cpu().numpy() # [100] aligned lambda values
            pred_dGs_for_system = pred_dGs_per_window_squeezed[i].cpu().numpy() # [100] predicted dGs

            # Reconstruct true dGs for 100 standard lambda windows
            true_dGs_for_system_aligned = np.full(100, np.nan, dtype=np.float32) # Use NaN for missing windows

            # Get original mu, sigma, and their 100-grid indices for the current system
            original_mu_targets = batch_all_original_mu_targets[i].cpu().numpy()
            original_sigma_targets = batch_all_original_sigma_targets[i].cpu().numpy() # These are std dev (sigma)
            original_100_grid_indices = batch_all_original_100_grid_indices[i].cpu().numpy()

            for j in range(len(original_mu_targets)): # Iterate through original (non-interpolated) windows for this system
                mu = original_mu_targets[j]
                sigma = original_sigma_targets[j] # This is std dev, not variance
                # Calculate dG_n using mu and sigma^2/2. Assumes sigma is standard deviation.
                true_dG_val = mu - (sigma**2 / 2)

                grid_idx = original_100_grid_indices[j]
                if 0 <= grid_idx < 100: # Ensure index is valid and within bounds
                    true_dGs_for_system_aligned[grid_idx] = true_dG_val
            
            all_systems_labels_for_windows.append(current_system_label)
            all_systems_aligned_lambdas.append(aligned_lambdas)
            all_systems_pred_dGs_per_window.append(pred_dGs_for_system)
            all_systems_true_dGs_per_window_aligned.append(true_dGs_for_system_aligned)

    metric_logger.synchronize_between_processes()

    print(f'system_labels:{system_labels}') # This print might show the labels of the last batch only
    # Print averaged losses
    print('Validation averaged total dG loss: {total_dg_losses.global_avg:.3f}'.format(total_dg_losses=metric_logger.total_dg_loss))
    print('Validation averaged aggregated dG loss: {agg_dg_losses.global_avg:.3f}'.format(agg_dg_losses=metric_logger.agg_dg_loss))
    print('Validation averaged smoothness loss: {smoothness_losses.global_avg:.3f}'.format(smoothness_losses=metric_logger.smoothness_loss))
    print('Validation averaged feature loss: {feature_losses.global_avg:.3f}'.format(feature_losses=metric_logger.feature_loss)) # Print avg feature loss
    print('Validation averaged combined loss: {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('Validation averaged MAE: {mean_abs_error.global_avg:.3f}'.format(mean_abs_error=metric_logger.mean_abs_error)) # Print avg MAE

    # Print comparison of predicted vs true total dGs
    print("\nTotal dG Prediction vs True Value:")
    # Ensure lists are of the same length before zipping
    min_len = min(len(pred_total_dGs_list), len(true_total_dGs_list), len(all_system_labels))
    for pred, true, label in zip(pred_total_dGs_list[:min_len], true_total_dGs_list[:min_len], all_system_labels[:min_len]):
        print(f"System {label}: Predicted_dG = {pred:.2f}, True_dG = {true:.2f}, Error = {abs(pred - true):.2f}")

    # Save results to CSV file
    def save_validation_results(pred_dGs, true_dGs, labels, save_path):
        # Ensure lists are of the same length
        min_len = min(len(pred_dGs), len(true_dGs), len(labels))
        pred_dGs = [float(p) for p in pred_dGs[:min_len]]  # Convert numpy floats to python floats
        true_dGs = [float(t) for t in true_dGs[:min_len]]

        results_df = pd.DataFrame({
            'System': labels[:min_len],
            'Predicted_dG': pred_dGs,
            'True_dG': true_dGs,
            'Error': [abs(p - t) for p, t in zip(pred_dGs, true_dGs)]
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nvalidation results saved: {save_path}")

    # In validate function end, call save
    save_path = '/nfs/export4_25T/ynlu/MAE-test/test_tool/output_dir/validation_results.csv' # Use a dedicated path for validation results
    save_validation_results(pred_total_dGs_list, true_total_dGs_list, all_system_labels, save_path)

    # Call the new saving function for per-window dG results
    per_window_save_path = os.path.join(os.path.dirname(save_path), 'validation_per_window_dG_for_plot.csv')
    save_per_window_dG_results_for_plot(
        all_systems_labels_for_windows,
        all_systems_aligned_lambdas,
        all_systems_pred_dGs_per_window,
        all_systems_true_dGs_per_window_aligned,
        per_window_save_path
    )

    return {
        'loss': float(metric_logger.loss.global_avg),  # Combined loss
        'total_dg_loss': float(metric_logger.total_dg_loss.global_avg),
        'agg_dg_loss': float(metric_logger.agg_dg_loss.global_avg),
        'smoothness_loss': float(metric_logger.smoothness_loss.global_avg),
        'feature_loss': float(metric_logger.feature_loss.global_avg), # Return avg feature loss
        'mean_abs_error': float(metric_logger.mean_abs_error.global_avg), # Return avg Mean Absolute Error (MAE)
    }