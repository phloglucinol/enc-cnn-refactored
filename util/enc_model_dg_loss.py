import torch
import torch.nn as nn


def dg_aggregation_loss_v2(pred_dGs: torch.Tensor,
                           original_window_dGs_list: list[torch.Tensor], # List of original window dGs (variable length, kcal/mol)
                           original_dlambda_list: list[float],           # List of original delta_lambda (from filenames)
                           reduction: str = 'mean'):
    """
    计算预测的 ΔG (Δλ=0.01, 无 kbt) 与目标 ΔG (可变 Δλ, kcal/mol) 之间的损失 (v2).

    函数内部会将模型预测的 ΔG_n (无 kbt) 乘以 kbt (0.592) 转换为 kcal/mol 单位，
    然后与目标 ΔG (kcal/mol) 进行比较计算损失。

    Args:
        pred_dGs (torch.Tensor): 模型预测的 ΔG_n 值 (无 kbt). 形状 (batch_size, 100).
                                  对应 Δλ=0.01 的窗口 [0.00, 0.01], ..., [0.99, 1.00].
        original_window_dGs_list (list[torch.Tensor]): 批次中每个样本对应的原始体系
                                                      的窗口 ΔG 列表 (kcal/mol). 每个 Tensor 形状 [num_original_windows].
        original_dlambda_list (list[float]): 批次中每个样本对应的原始体系的 Δλ 列表 (从文件名解析).
        reduction (str): 指定要应用的 reduce 方法:
                         'none' | 'mean' | 'sum'. 默认: 'mean'.

    Returns:
        torch.Tensor: 计算出的损失值. 根据 reduction 参数返回标量或形状为 (batch_size,) 的张量 (当 reduction='none' 时).
    """
    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of 'none', 'mean', or 'sum'.")

    batch_size = pred_dGs.shape[0]
    if len(original_window_dGs_list) != batch_size or len(original_dlambda_list) != batch_size:
         raise ValueError("Batch sizes of pred_dGs, original_window_dGs_list, and original_dlambda_list must match.")

    # Apply KBT to model predictions to match the scale of original_window_dGs (kcal/mol)
    kbt = 0.592 # Consistent KBT value
    pred_dGs_kcal_mol = pred_dGs * kbt # [batch_size, 100]

    # Calculate cumulative sum of the KBT-scaled predictions
    # cumulative_pred_dGs_kcal_mol[i, k] = sum(pred_dGs_kcal_mol[i, 0:k]) for k > 0, and 0 for k=0.
    cumulative_pred_dGs_kcal_mol = torch.cat([torch.zeros_like(pred_dGs_kcal_mol[:, :1]), pred_dGs_kcal_mol], dim=1).cumsum(dim=1) # [batch_size, 101]

    losses_per_sample = []

    for i in range(batch_size):
        sample_pred_cumulative = cumulative_pred_dGs_kcal_mol[i] # [101]
        # Target dGs are already in kcal/mol as read from free_ene.csv
        sample_target_dGs = original_window_dGs_list[i].to(pred_dGs.device) # [num_original_windows]
        sample_target_dlambda = original_dlambda_list[i] # float (from filename parsing)

        # 计算目标 Δλ 相对于预测 Δλ=0.01 的倍数。这用于确定一个目标窗口覆盖多少个预测窗口
        # 使用 round 是为了处理浮点数精度问题
        # Ensure we don't divide by zero if sample_target_dlambda is 0 (unlikely but safe check)
        if sample_target_dlambda is None or sample_target_dlambda <= 0:
            print(f"Warning: sample_target_dlambda is None or non-positive ({sample_target_dlambda}) for sample {i}. Cannot calculate ratio. Skipping sample loss.")
            # Append 0 loss for this sample, or handle as error
            losses_per_sample.append(torch.tensor(0.0, device=pred_dGs.device))
            continue # Skip to next sample

        # Calculate ratio of target dlambda to prediction dlambda (0.01)
        ratio = round((sample_target_dlambda / 0.01))

        # Add checks for extreme or zero ratio that would cause issues
        if ratio <= 0:
            print(f"Warning: Calculated ratio is non-positive ({ratio}) for sample {i} with target_dlambda {sample_target_dlambda}. Setting ratio to 1.")
            ratio = 1
        if ratio > 100: # A ratio > 100 means a single target window is > 1.0 lambda range, which is unusual.
            print(f"Warning: Calculated ratio is very large ({ratio}) for sample {i} with target_dlambda {sample_target_dlambda}. Capping ratio at 100.")
            ratio = 100

        num_original_windows = sample_target_dGs.shape[0]
        aggregated_pred_dGs = []

        # 聚合预测 ΔG 到目标窗口数量
        for j in range(num_original_windows):
            # 目标窗口 j (从 0 开始) 对应的 λ 范围是 [j * sample_target_dlambda, (j+1) * sample_target_dlambda)
            # 目标窗口 j 对应于预测窗口 (Δλ=0.01) 的索引范围 [j * ratio : (j+1) * ratio)
            # These indices are used to access the cumulative sum array (size 101, indices 0-100)
            start_idx_for_cumsum = j * ratio
            end_idx_for_cumsum = (j + 1) * ratio

            # Capping indices to be within the bounds of the cumulative sum tensor [0, 100]
            # min(..., 100) ensures indices don't go past the last valid index (100)
            start_idx_for_cumsum = min(start_idx_for_cumsum, 100)
            end_idx_for_cumsum = min(end_idx_for_cumsum, 100) # Cap end index at 100

            # Ensure end index is always >= start index after capping (unlikely with positive ratio, but safe)
            end_idx_for_cumsum = max(end_idx_for_cumsum, start_idx_for_cumsum)

            # Skip aggregation if the window range is empty after capping (can happen if start_idx == end_idx)
            if start_idx_for_cumsum == end_idx_for_cumsum:
                # This means the target window maps to zero predicted 0.01 windows.
                # This might indicate data issues or a very small target dlambda rounded to 0.
                # Assigning 0 aggregated value seems reasonable in this edge case.
                print(f"Warning: Aggregation range is empty for sample {i}, target window {j}. start={start_idx_for_cumsum}, end={end_idx_for_cumsum}. Aggregated value will be 0.")
                aggregated_value = torch.tensor(0.0, device=sample_pred_cumulative.device)
            else:
                # Use the capped indices for the cumulative sum
                # Use cumulative sum to get the sum of pred_dGs_kcal_mol from index start_idx_for_cumsum to end_idx_for_cumsum - 1
                # Sum arr[a:b] = cum_arr[b] - cum_arr[a] where cum_arr[k] = sum(arr[0:k])
                # Our cumulative_pred_dGs_kcal_mol[k] = sum(padded_pred_dGs_kcal_mol[0:k])
                # So sum of pred_dGs_kcal_mol[start_idx_for_cumsum : end_idx_for_cumsum] (indices 0-99 in the original pred_dGs_kcal_mol)
                # is cumulative_pred_dGs_kcal_mol[end_idx_for_cumsum] - cumulative_pred_dGs_kcal_mol[start_idx_for_cumsum].
                aggregated_value = sample_pred_cumulative[end_idx_for_cumsum] - sample_pred_cumulative[start_idx_for_cumsum]

            aggregated_pred_dGs.append(aggregated_value)

        # Check if any aggregated values were collected (should match num_original_windows unless skipped)
        if len(aggregated_pred_dGs) != num_original_windows:
            print(f"Error: Number of aggregated prediction values ({len(aggregated_pred_dGs)}) does not match number of target windows ({num_original_windows}) for sample {i}. Skipping sample loss.")
            losses_per_sample.append(torch.tensor(0.0, device=pred_dGs.device)) # Append 0 loss or NaN
            continue

        # 将该样本的聚合结果堆叠成张量
        aggregated_pred_dGs_tensor = torch.stack(aggregated_pred_dGs) # 形状 [num_original_windows]

        # 计算该样本的平方误差 (predictions and targets are both in kcal/mol)
        # 确保形状匹配 - 聚合后的预测形状应该与样本的原始窗口 dGs 形状相同
        # Shape check should already pass if the previous check passed, but leaving for safety
        if aggregated_pred_dGs_tensor.shape != sample_target_dGs.shape:
            # This is a critical check. If the shapes don't match, the aggregation logic or data is wrong.
            # This might indicate an issue with num_original_windows calculation or padding.
            print(f"Runtime Error: Shape mismatch after aggregation for sample {i}: aggregated_pred_dGs_tensor {aggregated_pred_dGs_tensor.shape}, sample_target_dGs {sample_target_dGs.shape}. Appending 0 loss.")
            losses_per_sample.append(torch.tensor(0.0, device=pred_dGs.device)) # Append 0 loss
            continue # Skip this sample

        sample_loss = (aggregated_pred_dGs_tensor - sample_target_dGs) ** 2 # [num_original_windows]
        losses_per_sample.append(sample_loss.mean()) # 计算每个样本的平均窗口损失

    # Check if any sample losses were collected
    if not losses_per_sample:
        print("Warning: No sample losses were collected in dg_aggregation_loss_v2.")
        # Return 0 or NaN depending on desired behavior for empty batches/errors
        if reduction == 'none':
            return torch.empty(0, device=pred_dGs.device)
        else:
            return torch.tensor(0.0, device=pred_dGs.device)

    # 将每个样本的平均损失堆叠起来
    batch_losses = torch.stack(losses_per_sample) # [batch_size]

    # 应用最终 reduction
    if reduction == 'none':
        return batch_losses # 返回每个样本的平均窗口损失
    elif reduction == 'mean':
        # 计算批次中所有样本平均损失的平均值
        return torch.mean(batch_losses)
    elif reduction == 'sum':
        # 计算批次中所有样本平均损失的总和
        return torch.sum(batch_losses)


# --- 使用示例 (使用列表输入) ---
if __name__ == '__main__':
    # 模拟数据
    batch_size = 4
    # Simulate model prediction of dG_n (without kbt)
    pred_dGs_sim = torch.randn(batch_size, 100) * 0.1 # Assume this is dG_n, stddev 0.1

    # Simulate original system window dGs (kcal/mol) and Δλ lists (variable length)
    original_window_dGs_list_sim = []
    original_dlambda_list_sim = []

    # Sample 0: Δλ=0.02 (50 windows)
    num_windows_0 = 50
    # Simulate target dGs in kcal/mol
    original_window_dGs_list_sim.append(torch.randn(num_windows_0) * 0.2 + 1.0) # Simulate mean 1.0, std 0.2 kcal/mol
    original_dlambda_list_sim.append(0.02)

    # Sample 1: Δλ=0.04 (25 windows)
    num_windows_1 = 25
    original_window_dGs_list_sim.append(torch.randn(num_windows_1) * 0.3 + 2.0) # Simulate mean 2.0, std 0.3 kcal/mol
    original_dlambda_list_sim.append(0.04)

    # Sample 2: Δλ=0.01 (100 windows)
    num_windows_2 = 100
    original_window_dGs_list_sim.append(torch.randn(num_windows_2) * 0.1 + 0.5) # Simulate mean 0.5, std 0.1 kcal/mol
    original_dlambda_list_sim.append(0.01)

    # Sample 3: Δλ=0.05 (20 windows)
    num_windows_3 = 20
    original_window_dGs_list_sim.append(torch.randn(num_windows_3) * 0.4 + 3.0) # Simulate mean 3.0, std 0.4 kcal/mol
    original_dlambda_list_sim.append(0.05)


    print("\nTesting dg_aggregation_loss_v2 (with list inputs, targets in kcal/mol)...")

    # Calculate loss (MSE reduction='mean')
    # Pass pred_dGs (without kbt) and targets (in kcal/mol)
    loss_mean_v2 = dg_aggregation_loss_v2(pred_dGs_sim, original_window_dGs_list_sim, original_dlambda_list_sim, reduction='mean')
    print(f"Loss v2 (mean): {loss_mean_v2.item()}")

    # Calculate loss (MSE reduction='none')
    loss_none_v2 = dg_aggregation_loss_v2(pred_dGs_sim, original_window_dGs_list_sim, original_dlambda_list_sim, reduction='none')
    print(f"Loss v2 (none) shape: {loss_none_v2.shape}") # Should be [batch_size]
    print(f"Loss v2 (none) per sample: {loss_none_v2.tolist()}")

    # Note: The test samples' target dGs are random, so losses will be non-zero.
    # In a real scenario, original_window_dGs_list_sim would come from actual free_ene.csv processing.
