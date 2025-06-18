import os
from natsort import natsorted
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

import torch
from scipy.interpolate import interp1d
import torch.nn as nn
from typing import List, Dict, Sequence


# data预处理模块
class LambdaDataProcessor:
    def __init__(self, min_delta=0.01, max_windows=100, max_data=100, include_delta_in_data=True):
        """
        :param min_delta: 补全窗口的Δλ值（固定0.01）
        :param max_windows: 目标标准网格的总窗口数（例如 100）
        :param max_data: 目标标准网格的每窗口最大数据点数（例如 40 或 100）
        """
        self.include_delta_in_data = include_delta_in_data
        self.min_delta = min_delta
        self.max_windows = max_windows
        self.max_data = max_data

        # 若需要λ=1.00，应调整max_windows=101
        # 标准 λ 网格，max_windows 个点：[0.00, 0.01, 0.02, ..., 0.99]  # 共 100 个点
        self.std_lambdas = np.round(np.arange(0, 1.0, min_delta), 4)[:max_windows]  # λ从0.00到0.99

    # --------------------------------------------------------
    def process(self, original_data: Sequence[Sequence[torch.Tensor]],
                original_lambdas: Sequence[Sequence[float]],
                original_deltas: Sequence[Sequence[float]],
                original_data_lengths: Sequence[Sequence[int]]) -> Dict:
        """
        返回修改后的字典结构：
        {
            'data': torch.Tensor [N, C', max_windows, max_data], # C' is 3 or 4 based on include_delta_in_data
            'lambdas': torch.Tensor [N, max_windows],            # λ值，空缺处填充标准λ
            'deltas': torch.Tensor [N, max_windows],             # Δλ值，空缺处填充-1
            'masks': {
                'window': torch.Tensor [N, max_windows],        # 有效窗口标记
                'delta': torch.Tensor [N, max_windows]          # 真实Δλ标记（非补全）
            },
            'original_lengths': torch.Tensor [N, max_windows],     # 原始数据长度
        }
        """
        batch_size = len(original_data)
        if batch_size == 0:
            # Determine output channels based on config
            num_output_channels = 4 if self.include_delta_in_data else 3
            return {
                'data': torch.empty(0, num_output_channels, self.max_windows, self.max_data),
                'lambdas': torch.empty(0, self.max_windows),
                'deltas': torch.empty(0, self.max_windows),
                'masks': {
                    'window': torch.empty(0, self.max_windows),
                    'delta': torch.empty(0, self.max_windows)
                },
                'original_lengths': torch.empty(0, self.max_windows)
            }

        device = 'cpu'
        if batch_size > 0 and len(original_data[0]) > 0 and isinstance(original_data[0][0], torch.Tensor):
            device = original_data[0][0].device

        # Determine expected input channels for original_data
        expected_input_channels = 3
        # Determine output channels for aligned_data
        num_output_channels = 4 if self.include_delta_in_data else 3


        # 初始化输出张量
        aligned_data = torch.zeros(batch_size, num_output_channels, self.max_windows, self.max_data, device=device)
        aligned_lambdas = torch.zeros(batch_size, self.max_windows, device=device)
        aligned_deltas = torch.full((batch_size, self.max_windows), -1.0, device=device)  # 空缺处填充-1

        masks = {
            'window': torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device),
            'delta': torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device)
        }

        std_lambdas_tensor = torch.from_numpy(self.std_lambdas).to(dtype=torch.float32, device=device)

        for i in range(batch_size):
            lambdas = original_lambdas[i]
            deltas = original_deltas[i]
            sample_raw_data = original_data[i] # List of tensors, shape [C_in, num_points]
            sample_data_lengths = original_data_lengths[i]

            # 输入验证（与之前相同）
            num_windows_actual = len(lambdas)
            assert num_windows_actual == len(
                deltas), f"样本 {i}: λ 数量 ({num_windows_actual}) 与 Δλ 数量 ({len(deltas)}) 不匹配"
            assert num_windows_actual == len(
                sample_raw_data), f"样本 {i}: λ/Δλ 数量 ({num_windows_actual}) 与 原始数据张量列表长度 ({len(sample_raw_data)}) 不匹配"
            assert num_windows_actual == len(
                sample_data_lengths), f"样本 {i}: λ/Δλ 数量 ({num_windows_actual}) 与 数据点数列表长度 ({len(sample_data_lengths)}) 不匹配"

            for orig_idx in range(num_windows_actual):
                lbda = lambdas[orig_idx]
                delta = deltas[orig_idx]
                current_window_data = sample_raw_data[orig_idx].to(device) # Shape [C_in, num_points]
                current_data_length = sample_data_lengths[orig_idx]

                # --- MODIFIED ASSERTION ---
                # Assert that the input data channels match the expected channels based on include_delta_in_data config
                expected_input_channels_for_this_sample = 4 if self.include_delta_in_data else 3
                assert current_window_data.shape[
                           0] == expected_input_channels_for_this_sample, \
                           f"样本 {i}, 原始窗口 {orig_idx}: 数据张量通道数应为 {expected_input_channels_for_this_sample}, 实际为 {current_window_data.shape[0]}. 请检查数据加载或生成逻辑是否与 include_delta_in_data ({self.include_delta_in_data}) 配置匹配。"
                # --- END MODIFIED ASSERTION ---

                assert current_window_data.shape[
                           1] == current_data_length, f"样本 {i}, 原始窗口 {orig_idx}: 数据张量点数 ({current_window_data.shape[1]}) 与报告长度 ({current_data_length}) 不匹配"

                num_points_to_copy = min(current_data_length, self.max_data)
                if current_data_length > self.max_data:
                    print(
                        f"警告: 样本 {i}, 原始窗口 {orig_idx} (λ={lbda:.2f}) 的数据点数 ({current_data_length}) 超过 max_data ({self.max_data}). 数据将被截断.")
                    # Truncate the raw data BEFORE copying
                    current_window_data = current_window_data[:, :self.max_data]


                target_idx = np.abs(self.std_lambdas - lbda).argmin()
                if target_idx < self.max_windows:
                    # --- MODIFIED DATA COPYING ---
                    # Copy channels based on the number of output channels required by the processor
                    # If num_output_channels is 3, copy only the first 3 channels from input
                    # If num_output_channels is 4, copy all 4 channels from input (assuming input is already 4 channels)
                    channels_to_copy = min(current_window_data.shape[0], num_output_channels) # Copy up to the available channels, but no more than needed for output
                    aligned_data[i, :channels_to_copy, target_idx, :num_points_to_copy] = current_window_data[:channels_to_copy, :num_points_to_copy]

                    # Note: If include_delta_in_data is True, the 4th channel (delta) is copied directly from the 4th channel of current_window_data.
                    # The separate delta handling block below is only needed if include_delta_in_data is False,
                    # but we still want to store deltas in aligned_deltas. The current logic handles both cases.
                    # --- END MODIFIED DATA COPYING ---


                    # 记录λ和Δλ（单独存储）
                    aligned_lambdas[i, target_idx] = lbda
                    aligned_deltas[i, target_idx] = delta  # 有效Δλ覆盖-1

                    # 设置掩码
                    masks['window'][i, target_idx] = 1
                    masks['delta'][i, target_idx] = 1

                    # --- REMOVED REDUNDANT DELTA COPYING ---
                    # This block is redundant because if include_delta_in_data is True,
                    # the 4th channel was already copied in the MODIFIED DATA COPYING section above.
                    # If include_delta_in_data is False, we don't want delta in aligned_data anyway.
                    # if self.include_delta_in_data:
                    #     # 将当前窗口的delta值广播到第4通道
                    #     # This assumes the original data *didn't* already have the delta channel,
                    #     # which contradicts how the test script now generates data.
                    #     aligned_data[i, 3, target_idx, :num_points_to_copy] = delta
                    # --- END REMOVED ---


            # 用标准λ填充空缺位置（对齐后的λ始终完整）
            aligned_lambdas[i] = std_lambdas_tensor

            # --- MODIFIED DELTA CHANNEL FILLING ---
            # This logic correctly fills the delta channel (index 3) of aligned_data
            # IF self.include_delta_in_data is True.
            # For valid windows (mask['delta']==1), it should already have the real delta copied from the input data's 4th channel.
            # For padded windows (mask['delta']==0), aligned_deltas[i] will have -1.
            # We need to use aligned_deltas[i] to fill the 4th channel of aligned_data for *all* 100 windows.
            if self.include_delta_in_data:
                # aligned_deltas[i] contains original delta for valid windows and -1 for padded windows [max_windows]
                # Expand to [max_windows, max_data]
                delta_channel_data = aligned_deltas[i].unsqueeze(1).expand(-1, self.max_data)
                # Assign this to the 4th channel (index 3) of aligned_data
                # This overwrites the padded areas (where delta was not copied from input) with -1,
                # and keeps the original delta for valid windows (which was already copied from input).
                # This ensures the 4th channel of aligned_data consistently represents the delta for each of the 100 windows.
                aligned_data[i, 3, :, :] = delta_channel_data
            # --- END MODIFIED DELTA CHANNEL FILLING ---


        result = {
            'data': aligned_data,  # [N, C_out, 100, 50]
            'lambdas': aligned_lambdas,  # [N, 100]
            'deltas': aligned_deltas,  # [N, 100]（有效处为原始Δλ，空缺处为-1）
            'masks': masks
        }

        # 添加原始长度信息 - 这是关键的新增部分
        original_lengths_tensor = torch.zeros(batch_size, self.max_windows, dtype=torch.float32, device=device)

        for i in range(batch_size):
            sample_data_lengths = original_data_lengths[i]
            lambdas = original_lambdas[i]

            # 记录每个窗口的原始数据点数量
            for orig_idx in range(len(lambdas)):
                lbda = lambdas[orig_idx]
                orig_length = sample_data_lengths[orig_idx]

                # 找到标准网格中最接近的λ索引
                target_idx = np.abs(self.std_lambdas - lbda).argmin()
                if target_idx < self.max_windows:
                    original_lengths_tensor[i, target_idx] = orig_length

        # 将原始长度添加到返回结果
        result['original_lengths'] = original_lengths_tensor

        return result

def natural_sort_key(s):
    """
    自然排序的辅助函数，用于提取文件名中的数字部分
    [0-9]+：匹配一个或多个连续的数字。
    ()：捕获分组，表示匹配的内容会被保留在拆分结果中。
    re.split：根据正则表达式拆分字符串，并返回一个列表。
    """
    # re.split('([0-9]+)', s)将字符串 s 按照数字部分（[0-9]+）进行拆分。特别之处在于，通过将正则表达式用括号 () 包裹，拆分后的结果会保留数字部分。
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# NOTE: build_dataset might need dg_root_dir argument if used directly
# def build_dataset(is_train, args):
#     root_dir = os.path.join(args.data_path, 'train' if is_train else 'val')
#     dataset = CustomDataset(root_dir) # Missing dg_root_dir here

#     print(dataset)

#     return dataset

# 解除注释并修改函数签名 - 移除 dg_root_dir
def build_dataset(is_train, args):
    # 使用 args.data_path 构建原始数据 root_dir
    root_dir = os.path.join(args.data_path, 'train' if is_train else 'val')
    # 实例化 CustomDataset 时传递一个 root 目录
    dataset = CustomDataset(root_dir) # 只传递一个 root_dir

    print(dataset)

    return dataset


# We will define how to calculate total dG from raw data
# NOTE: This function is likely no longer needed if we read dG from free_ene.csv
# However, it provides context for the expected unit/calculation of total dG.
# Keeping it commented or for potential future use/validation.
# def calculate_total_dg(raw_data_list: Sequence[torch.Tensor]):
#     """
#     Calculates the total dG for a system from a list of raw window tensors.
#     raw_data_list: List of torch.Tensor, where each tensor is [3, num_points] for a window.
#                    Features are assumed to be [mu, sigma, error] along dim 1.
#     Returns: total dG (float in kcal/mol)
#     """
#     total_dG = 0.0
#     kbt = 0.592
#     for window_data in raw_data_list:
#         # Assuming features are [mu, sigma, error]
#         mu = window_data[0, -1]
#         sigma = window_data[1, -1]
#         error = window_data[2, -1]
#
#         # Convert sigma to sigma^2
#         sigma_sq = sigma ** 2
#
#         # Calculate dG_n for this window (without kbt)
#         dG_n_per_window = mu - (sigma_sq / 2.0) + error
#
#         # Add the Python float value of the window sum to the total dG (with kbt applied here)
#         # Use .item() to extract the scalar value from the tensor
#         total_dG += dG_n_per_window.item() * kbt  # 将每个窗口最后一个点的 (μ - σ²/2 + error) * kbt 相加
#
#     return total_dG  # This will now be a float in kcal/mol


class CustomDataset(Dataset):
    # 修改函数签名 - 移除 dg_root_dir 参数
    # 添加 num_random_subsets_per_system 参数
    def __init__(self, root_dir, subset_size=10, processor_max_windows=100, processor_max_data=100, processor_include_delta=True,
                 num_random_subsets_per_system=100, # Add this new parameter with a default value, e.g., 100
                 per_lambda_max_points=None): # 新增参数，默认 None，表示不限制，或设置为processor_max_data
        """
        初始化数据集，加载原始体系数据和目标ΔG信息，生成子集索引。
        :param root_dir: 原始数据 (CSV) 和 ΔG 信息 (free_ene.csv) 的根目录，包含system_X文件夹
        :param subset_size: 每个子集包含的窗口数量 (e.g., 10)
        :param processor_max_windows: LambdaDataProcessor的目标窗口数 (e.g., 100)
        :param processor_max_data: LambdaDataProcessor的每窗口最大数据点数 (e.g., 100)
        :param processor_include_delta: Whether to include delta in processor output
        :param num_random_subsets_per_system: 从每个原始体系中随机采样的子集数量 (K)
        :param per_lambda_max_points: 每个lambda窗口最多采样的数据点数K (取前K个)。如果为None或大于max_data，则使用max_data。
        """
        self.root_dir = root_dir
        self.subset_size = subset_size
        self.processor_max_windows = processor_max_windows
        self.processor_max_data = processor_max_data
        self.processor_include_delta = processor_include_delta
        self.num_random_subsets_per_system = num_random_subsets_per_system # Store the new parameter
        # 确保 per_lambda_max_points 不超过 processor_max_data
        self.per_lambda_max_points = processor_max_data if per_lambda_max_points is None else min(per_lambda_max_points, processor_max_data)

        self.lambda_processor = LambdaDataProcessor(max_windows=self.processor_max_windows,
                                                    max_data=self.processor_max_data,
                                                    include_delta_in_data=processor_include_delta
                                                    )

        self.original_systems = [] # Stores info for each original system
        self.subset_indices = []   # Stores indices for all generated subsets
        self.system_target_dg = [] # Stores the total dG for each original system (Read from free_ene.csv, kcal/mol)

        # KBT constant for potential use, though conversion now happens in loss
        # self.kbt = 0.592 # Not strictly needed in dataset init anymore

        # 添加路径检查
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Data root directory '{root_dir}' does not exist.")

        # Get system folders from the root data directory
        folder_names = natsorted(os.listdir(root_dir))

        system_idx_counter = 0
        # 遍历所有文件夹（代表原始体系）
        for folder_name in folder_names:
            # Base path for this system within the root_dir
            system_base_path = os.path.join(root_dir, folder_name)

            if os.path.isdir(system_base_path):
                # 处理 complex 和 ligand 子文件夹
                for sub_folder_name in ['complex', 'ligand']:
                    # Path for raw data CSVs and the parent for fe_cal_out
                    sub_folder_path = os.path.join(system_base_path, sub_folder_name)
                    data_csv_path = sub_folder_path # Raw data CSVs are directly in sub_folder_path

                    # Path for target dG file (free_ene.csv) - Modified path construction
                    free_ene_path = os.path.join(sub_folder_path, 'fe_cal_out', 'free_ene_zwanzig.csv') # Corrected path based on new structure

                    if not os.path.isdir(data_csv_path):
                        # print(f"Warning: Data CSV directory '{data_csv_path}' not found. Skipping system part.") # Less noisy
                        continue # Skip if raw data path doesn't exist

                    if not os.path.exists(free_ene_path):
                        # print(f"Warning: free_ene.csv not found at '{free_ene_path}'. Skipping system part.") # Less noisy
                        continue # Skip if free_ene.csv doesn't exist

                    # Load all windows for this original system (raw data)
                    raw_data_system = []       # List[Tensor[3, num_points]]
                    lambdas_system = []        # List[float]
                    deltas_system = []         # List[float] # These will be the dlambda parsed from filenames
                    data_lengths_system = []   # List[int]

                    file_names = sorted([f for f in os.listdir(data_csv_path) if f.endswith('.csv')], key=natural_sort_key)

                    # Assuming each CSV corresponds to a lambda window
                    for file_name in file_names:
                        file_path = os.path.join(data_csv_path, file_name)
                        try:
                           data_np = pd.read_csv(file_path).values
                        except Exception as e:
                           print(f"Error reading CSV file {file_path}: {e}. Skipping file.")
                           continue

                        window_data_3chan = torch.tensor(data_np.T, dtype=torch.float32)  # Shape [3, num_points]
                        # Ensure sigma (channel 1) is non-negative before potentially adding delta channel
                        # This is a good practice if sigma is expected to be non-negative
                        # window_data_3chan[1, :] = torch.relu(window_data_3chan[1, :]) # Optional: Ensure sigma >= 0

                        # Attempt to extract lambda and delta (dlambda) from filename - Keep this logic
                        # Modified regex to better match float numbers without trailing periods
                        match = re.search(r'_lambda(\d+\.?\d*)_delta(\d+\.?\d*)', file_name)
                        parsed_lambda = None
                        parsed_delta = None
                        if match:
                            try:
                                parsed_lambda = float(match.group(1))
                                parsed_delta = float(match.group(2))
                            except ValueError as e:
                                print(f"Error converting lambda or delta to float in file {file_name}: {e}. Skipping file.")
                                continue # Skip this file

                        if parsed_lambda is None or parsed_delta is None:
                             print(f"Warning: Could not parse lambda/delta from filename {file_name}. Skipping file.")
                             continue # Skip this file if parsing failed

                        # --- Add Delta Channel if Required ---
                        if self.processor_include_delta:
                             # Add delta as the 4th channel, broadcast to data_length
                             data_length = window_data_3chan.shape[1]
                             current_delta_tensor = torch.full((1, data_length), parsed_delta, dtype=torch.float32, device=window_data_3chan.device)
                             window_data = torch.cat([window_data_3chan, current_delta_tensor], dim=0) # Shape [4, data_length]
                        else:
                             window_data = window_data_3chan # Shape [3, data_length]
                        # --- End Add Delta Channel ---

                        raw_data_system.append(window_data)
                        data_lengths_system.append(window_data.shape[1])  # num_points
                        lambdas_system.append(parsed_lambda)
                        deltas_system.append(parsed_delta) # Store the parsed delta

                    if not raw_data_system:
                        print(f"Warning: No valid CSV files found or parsed in {data_csv_path}. Skipping system part.")
                        continue

                    # Ensure collected lists have consistent length after parsing/skipping
                    # No, this logic is incorrect. valid_indices logic should be applied here.
                    # Let's filter based on successful parsing and data availability
                    filtered_raw_data_system = []
                    filtered_lambdas_system = []
                    filtered_deltas_system = []
                    filtered_data_lengths_system = []
                    for i in range(len(raw_data_system)):
                        # Assume raw_data_system was only appended if parsing succeeded and data is valid
                        # The check 'if parsed_lambda is None or parsed_delta is None:' already handles skipping files.
                        # The check 'if not raw_data_system:' handles empty directories.
                        # No extra filtering based on length mismatches within this inner loop should be needed if the above checks are correct.
                        # So, the assumption was that raw_data_system, lambdas_system, deltas_system, data_lengths_system
                        # are built in sync, one item per valid file.
                        # If there are mismatches, it's better to catch them early or ensure file parsing is robust.
                        # For now, proceed assuming lists are in sync from successful file reads.
                        filtered_raw_data_system.append(raw_data_system[i])
                        filtered_lambdas_system.append(lambdas_system[i])
                        filtered_deltas_system.append(deltas_system[i])
                        filtered_data_lengths_system.append(data_lengths_system[i])

                    # This check ensures lists are consistent *after* loading files.
                    if not (len(filtered_raw_data_system) == len(filtered_lambdas_system) == len(filtered_deltas_system) == len(filtered_data_lengths_system)):
                        print(f"Error: Mismatch between number of windows ({len(filtered_raw_data_system)}) and parsed lists lengths after filtering in {data_csv_path}. Skipping system part.")
                        continue

                    raw_data_system = filtered_raw_data_system
                    lambdas_system = filtered_lambdas_system
                    deltas_system = filtered_deltas_system
                    data_lengths_system = filtered_data_lengths_system


                    # --- Read Target dG and Window dGs from free_ene.csv ---
                    try:
                        free_ene_df = pd.read_csv(free_ene_path, sep='|')

                        if free_ene_df.empty:
                            print(f"Warning: free_ene.csv is empty at '{free_ene_path}'. Skipping system part.")
                            continue

                        # Extract total dG from the last row, second column (kcal/mol)
                        total_dg_kcal_mol = free_ene_df.iloc[-1, 1]
                        self.system_target_dg.append(total_dg_kcal_mol) # Store directly (kcal/mol)

                        # Extract window dGs from rows excluding header and last row (kcal/mol)
                        window_dGs_kcal_mol = free_ene_df.iloc[:-1, 1].tolist()

                        # Get the system's true delta_lambda from the deltas parsed from filenames
                        # Assuming dlambda is consistent across windows for a given system part (complex/ligand)
                        system_dlambda = deltas_system[0] if deltas_system else 0.01 # Use delta from first filename as system dlambda

                    except Exception as e:
                        print(f"Error processing free_ene.csv at '{free_ene_path}': {e}. Skipping system part.")
                        continue

                    # Store original system data (as lists) and loaded targets
                    self.original_systems.append({
                        'raw_data': raw_data_system, # List[Tensor[3 or 4, num_points]]
                        'lambdas': lambdas_system,   # List[float] (from filenames)
                        'deltas': deltas_system,     # List[float] (dlambda from filenames)
                        'lengths': data_lengths_system, # List[int]
                        'label': folder_name + '_' + sub_folder_name, # Label for the original system
                        # Store window dGs loaded from free_ene.csv (kcal/mol)
                        'window_dGs': window_dGs_kcal_mol, # List of floats (kcal/mol)

                        # 注意: 这里假设一个体系的所有窗口的 Δλ 是相同的，所以取 deltas_system[0]。如果 Δλ 在同一个体系内变化，你需要更精确地存储每个窗口的 Δλ，或者根据文件名解析每个窗口的 Δλ。
                        # Store system's true delta_lambda parsed from filenames - Keep this
                        'dlambda': system_dlambda # Use dlambda from filenames
                    })

                    # --- Generate subset indices ---
                    system_window_count = len(raw_data_system)
                    # Ensure the number of raw data windows matches the number of window dGs read from free_ene.csv
                    # This check is crucial before generating subsets
                    if system_window_count != len(window_dGs_kcal_mol):
                        print(f"Error: Mismatch between number of data CSVs ({system_window_count}) and window dGs in free_ene.csv ({len(window_dGs_kcal_mol)}) for system {folder_name}/{sub_folder_name}. Skipping subset generation.")
                        continue

                    if system_window_count < self.subset_size:
                        print(f"Warning: System {folder_name}/{sub_folder_name} has only {system_window_count} windows, less than subset size {self.subset_size}. Skipping subset generation for this system.")
                        continue

                    # --- Implement Random Sampling ---
                    # Generate num_random_subsets_per_system random subsets
                    all_window_indices = list(range(system_window_count))
                    for k in range(self.num_random_subsets_per_system):
                         # Randomly sample subset_size indices without replacement
                         # Use numpy.random.choice for efficient sampling
                         current_subset_indices = np.random.choice(all_window_indices, size=self.subset_size, replace=False).tolist()
                         # Sort indices for consistency, though not strictly required by __getitem__
                         current_subset_indices.sort()

                         self.subset_indices.append({
                             'system_idx': system_idx_counter, # Link to the original system
                             'window_indices': current_subset_indices, # The randomly sampled indices
                             'subset_label': f"{folder_name}_{sub_folder_name}_subset_random_{k}" # Unique label
                         })
                    # --- End Random Sampling ---

                    system_idx_counter += 1 # Increment for the next original system (complex/ligand pair)


        print(f"Loaded {len(self.original_systems)} original systems.")
        # Update print statement to reflect random sampling
        print(f"Generated {len(self.subset_indices)} subset samples (subset size {self.subset_size}, {self.num_random_subsets_per_system} random subsets per system).")
        # Data standardization will be handled by LambdaDataProcessor per subset


    def __len__(self):
        """返回数据集的大小（生成的子集样本的数量）"""
        return len(self.subset_indices)

    def __getitem__(self, idx):
        """
        根据索引返回一个子集样本。
        每个样本是经过 LambdaDataProcessor 处理后的子集数据，以及原始体系的总 ΔG 目标和窗口 ΔG 目标。
        现在额外返回原始体系的标签，以及整个原始体系的所有原始窗口的目标 μ, σ 值及其在 100 窗口网格中的索引。
        :param idx: 子集样本索引
        :return: processed_data_dict (Dict from LambdaDataProcessor), original_system_window_dGs (Tensor), original_system_dlambda (float), target_total_dg (Tensor), system_label (str), all_original_mu_targets_tensor (Tensor), all_original_sigma_targets_tensor (Tensor), all_original_100_grid_indices_tensor (Tensor)
        """
        # Get info for this subset sample
        subset_info = self.subset_indices[idx]
        system_idx = subset_info['system_idx']
        window_indices = subset_info['window_indices']
        # subset_label = subset_info['subset_label'] # Optional: return label if needed

        # Get the original system data and targets
        original_system = self.original_systems[system_idx]
        # total_dg is stored in self.system_target_dg, indexed by system_idx
        target_total_dg = self.system_target_dg[system_idx] # This is the total dG (kcal/mol)

        # Original window dGs for the *entire* original system (used by agg_dg_loss_v2)
        # These are in kcal/mol as read from free_ene.csv
        original_system_window_dGs = torch.tensor(original_system['window_dGs'], dtype=torch.float32)

        # Original system's delta_lambda (used by agg_dg_loss_v2) - Comes from filename parsing
        original_system_dlambda = original_system['dlambda'] # This is the true delta_lambda for the system

        # Extract the raw data, lambdas, deltas, and lengths for the selected *subset* window indices
        raw_data_subset = []
        lambdas_subset = []
        deltas_subset = []
        lengths_subset = [] # This will now store the potentially truncated length
        
        for i in window_indices:
            window_raw_tensor = original_system['raw_data'][i] # Shape [C, original_num_points]
            
            # --- NEW: Truncate window data points to per_lambda_max_points ---
            effective_points = min(window_raw_tensor.shape[1], self.per_lambda_max_points)
            truncated_window_tensor = window_raw_tensor[:, :effective_points]
            # --- END NEW ---

            raw_data_subset.append(truncated_window_tensor)
            lambdas_subset.append(original_system['lambdas'][i])
            deltas_subset.append(original_system['deltas'][i])
            lengths_subset.append(effective_points) # Store the actual (truncated) length

        # Use LambdaDataProcessor to process this subset (treat as batch size 1 for the processor)
        # The processor expects Sequence[Sequence[...]], so wrap lists in another list.
        processed_data_dict = self.lambda_processor.process(
            [raw_data_subset],
            [lambdas_subset],
            [deltas_subset], # Pass deltas from filenames
            [lengths_subset] # Pass the potentially truncated lengths
        )

        # Squeeze the batch dimension (size 1) from processor output
        for key in processed_data_dict.keys():
            if isinstance(processed_data_dict[key], torch.Tensor):
                 processed_data_dict[key] = processed_data_dict[key].squeeze(0)
            elif isinstance(processed_data_dict[key], dict): # Handle nested masks dict
                 for sub_key in processed_data_dict[key].keys():
                      if isinstance(processed_data_dict[key][sub_key], torch.Tensor):
                           processed_data_dict[key][sub_key] = processed_data_dict[key][sub_key].squeeze(0)

        # Return the processed subset data dictionary and the original system's total dG target
        # The DataLoader will batch these up.
        # The output shape from __getitem__ should be consistent.
        # Let's return the processed dict and the target_dg
        # DataLoader will then yield batches like:
        # {
        #   'data': Tensor[N, 3, 100, max_data],
        #   'lambdas': Tensor[N, 100],
        #   'deltas': Tensor[N, 100],
        #   'masks': {'window': Tensor[N, 100], 'delta': Tensor[N, 100]},
        #   'original_lengths': Tensor[N, 100]
        # },
        # target_dg: Tensor[N]

        # --- NEW: Extract mu and sigma targets from *ALL* raw_data_system ---
        # Also find their corresponding indices in the 100-grid output
        all_original_mu_targets = []
        all_original_sigma_targets = []
        all_original_100_grid_indices = [] # Store indices in the 100-grid

        std_lambdas_np = self.lambda_processor.std_lambdas # numpy array

        # Iterate through ALL original windows for this system (using original_system data)
        for i, raw_tensor in enumerate(original_system['raw_data']):
             # Ensure tensor has enough channels and points
             # raw_tensor shape is [C_in, num_points] where C_in is 3 or 4
             expected_input_channels = 4 if self.processor_include_delta else 3
             if raw_tensor.shape[0] >= 3 and raw_tensor.shape[1] > 0: # Need at least 3 channels (mu, sigma, error)

                 # Extract mu and sigma from the last point
                 # Use indices 0 for mu, 1 for sigma
                 all_original_mu_targets.append(raw_tensor[0, -1].item()) # Extract scalar
                 all_original_sigma_targets.append(raw_tensor[1, -1].item()) # Extract scalar

                 # Find the closest standard lambda index for this original window's lambda
                 original_lambda = original_system['lambdas'][i] # Use lambda from ALL original windows
                 target_idx_100_grid = np.abs(std_lambdas_np - original_lambda).argmin()
                 # Ensure index is within bounds of the 100-grid
                 if target_idx_100_grid < self.processor_max_windows:
                    all_original_100_grid_indices.append(int(target_idx_100_grid)) # Store the index
                 else:
                    # This warning is less likely if std_lambdas cover the expected lambda range
                    # print(f"Warning: Calculated 100-grid index ({target_idx_100_grid}) out of bounds for original window index {i} (lambda {original_lambda}) in system {original_system['label']}. Max windows is {self.processor_max_windows}. Skipping for feature loss targets.")
                    # Do not append target if index is out of bounds
                    pass # Skip this window for feature loss targets - indices list and target lists must match size

             else:
                 # Handle insufficient data warning
                 # This case should ideally not happen if the initial CSV loading and filtering are correct
                 print(f"Warning: Insufficient data in raw_data_system tensor (shape {raw_tensor.shape}, expected >= [3, >0]) for system {original_system['label']}, original window index {i}. Cannot extract mu/sigma target or find 100-grid index. Skipping this window for feature loss targets.")
                 pass # Skip this window for feature loss targets - indices list and target lists must match size


        # Need to ensure the lists for mu, sigma, and indices are of the same length
        # if any were skipped due to the checks above.
        min_len_targets = min(len(all_original_mu_targets), len(all_original_sigma_targets), len(all_original_100_grid_indices))
        all_original_mu_targets = all_original_mu_targets[:min_len_targets]
        all_original_sigma_targets = all_original_sigma_targets[:min_len_targets]
        all_original_100_grid_indices = all_original_100_grid_indices[:min_len_targets]
        if len(all_original_mu_targets) != len(original_system['raw_data']):
             print(f"Warning: Skipped {len(original_system['raw_data']) - len(all_original_mu_targets)} original windows for feature targets in system {original_system['label']} due to data shape/index issues.")


        # Convert lists to tensors (these will be variable length per sample)
        all_original_mu_targets_tensor = torch.tensor(all_original_mu_targets, dtype=torch.float32) # Shape [num_original_windows_total]
        all_original_sigma_targets_tensor = torch.tensor(all_original_sigma_targets, dtype=torch.float32) # Shape [num_original_windows_total]
        all_original_100_grid_indices_tensor = torch.tensor(all_original_100_grid_indices, dtype=torch.long) # Shape [num_original_windows_total]


        # Return the processed subset data dictionary, the original system's *entire* window dGs tensor (kcal/mol),
        # the original system's delta_lambda (from filenames), the original system's total dG target (kcal/mol), and the system label
        # The DataLoader will batch these up using custom_collate_fn.
        # --- MODIFIED RETURN ---
        return (processed_data_dict,
                original_system_window_dGs,  # Pass as tensor (kcal/mol)
                original_system_dlambda,    # Pass as float (from filenames)
                torch.tensor(target_total_dg, dtype=torch.float32),  # Pass as tensor (kcal/mol)
                original_system['label'], # Pass label
                all_original_mu_targets_tensor, # Pass new mu targets tensor [num_original_windows_total]
                all_original_sigma_targets_tensor, # Pass new sigma targets tensor [num_original_windows_total]
                all_original_100_grid_indices_tensor # Pass new 100-grid indices tensor [num_original_windows_total]
               )
        # --- END MODIFIED RETURN ---


# --- custom_collate_fn function (Moved outside __main__) ---
def custom_collate_fn(batch):
    # batch is a list of tuples: [(processed_dict_1, original_window_dGs_1, ..., all_mu_targets_1, all_sigma_targets_1, all_indices_1), ...]
    processed_dicts = [item[0] for item in batch]
    original_window_dGs_list = [item[1] for item in batch]
    original_dlambda_list = [item[2] for item in batch]
    target_total_dgs = [item[3] for item in batch]
    system_labels = [item[4] for item in batch]

    # Collect lists of variable-length tensors for ALL original window targets/indices
    batch_all_original_mu_targets = [item[5] for item in batch] # List of Tensors [num_original_windows_total_1], ...
    batch_all_original_sigma_targets = [item[6] for item in batch] # List of Tensors [...]
    batch_all_original_100_grid_indices = [item[7] for item in batch] # List of Tensors [...]

    # Stack fixed-size parts (processed_dict and target_total_dgs) - this logic remains the same
    batched_processed_dict = {}
    if processed_dicts: # Handle empty batch gracefully
        for key in processed_dicts[0].keys():
            if isinstance(processed_dicts[0][key], torch.Tensor):
                batched_processed_dict[key] = torch.stack([d[key] for d in processed_dicts])
            elif isinstance(processed_dicts[0][key], dict): # Handle nested masks dict
                batched_processed_dict[key] = {}
                for sub_key in processed_dicts[0][key].keys():
                    if isinstance(processed_dicts[0][key][sub_key], torch.Tensor):
                         batched_processed_dict[key][sub_key] = torch.stack([d[key][sub_key] for d in processed_dicts])
            # Add other types if necessary
    else: # Return empty batched dict if input batch is empty
         # Determine output channels based on LambdaDataProcessor default config or pass it here
         # For simplicity, assuming a default structure or it's handled upstream.
         # A robust solution might require passing processor config or handling empty tensors later.
         # Let's assume default 3 channels and common max_windows/max_data for the shape placeholder
         default_channels = 3 # Or infer from model config?
         default_max_windows = 100 # Or infer from model config?
         default_max_data = 50 # Or infer from model config?
         batched_processed_dict = {
              'data': torch.empty(0, default_channels, default_max_windows, default_max_data),
              'lambdas': torch.empty(0, default_max_windows),
              'deltas': torch.empty(0, default_max_windows),
              'masks': {
                   'window': torch.empty(0, default_max_windows),
                   'delta': torch.empty(0, default_max_windows)
              },
              'original_lengths': torch.empty(0, default_max_windows)
         }


    # Stack target_total_dgs
    if target_total_dgs:
         batched_target_total_dg = torch.stack(target_total_dgs)
    else:
         batched_target_total_dg = torch.empty(0)


    # Keep original_window_dGs_list and original_dlambda_list as lists
    # Keep batch_all_original_mu_targets, batch_all_original_sigma_targets, batch_all_original_100_grid_indices as lists
    # These lists' lengths will be the batch size. Each item in the list is a tensor/float corresponding to a sample.

    # --- MODIFIED RETURN ---
    return (batched_processed_dict,
            original_window_dGs_list,  # Pass as list of tensors (kcal/mol)
            original_dlambda_list,    # Pass as list of floats (from filenames)
            batched_target_total_dg,  # Pass as batched tensor (kcal/mol)
            system_labels,            # Pass list of labels
            batch_all_original_mu_targets, # Pass list of tensors (variable length per sample)
            batch_all_original_sigma_targets, # Pass list of tensors (variable length per sample)
            batch_all_original_100_grid_indices # Pass list of tensors (variable length per sample)
           )
    # --- END MODIFIED RETURN ---


# --- End custom_collate_fn function ---


if __name__ == '__main__':

    # 创建数据集实例
    # Adjust root_dir and dg_root_dir to your actual data location
    # Assuming raw data is under data_path/train/system_X/complex or /ligand
    # Assuming free_ene.csv is under dg_data_path/train/system_X/complex or /ligand/electrostatics/sample_csv_data/fe_cal_out

    # Example root directory (adjust as needed) - Now only one root is needed
    data_root = "/nfs/export4_25T/ynlu/data/enc_cnn_dU_info_dataset/8-1-1/s0" # This should be the base directory like data_path

    # Need to specify train/val within this root if applicable, or pass the specific train/val path as root_dir
    # Let's assume data_root itself is the base, and CustomDataset expects data_root/system_X/...
    # If your actual data structure is data_root/train/system_X/... then pass os.path.join(data_root, 'train') as the root_dir

    # Example: Assuming data_root/train contains system_X folders
    train_root_dir = os.path.join(data_root, 'train') # Adjust this line based on your actual structure

    subset_size = 10
    processor_max_data = 50  # Assuming max_data in model input is 50
    # Specify the number of random subsets per system for testing
    num_random_subsets = 10 # Example: Generate 10 random subsets per system part
    test_per_lambda_max_points = 20 # 设置每个λ窗口最多取20个数据点

    # Pass only the single root_dir
    print(f"Attempting to load training data from root: {train_root_dir}")
    # Pass the new parameter num_random_subsets_per_system here
    dataset = CustomDataset(train_root_dir, subset_size=subset_size, # Pass only one root
                            processor_max_data=processor_max_data,
                            processor_include_delta=False,  # Set to True to test new scheme
                            num_random_subsets_per_system=num_random_subsets, # Pass the parameter
                            per_lambda_max_points=test_per_lambda_max_points # 新增参数传递
                            )

    print(f"Dataset size (number of subsets): {len(dataset)}")

    # Print loaded system info for verification
    print("\n--- Loaded Original System Info (Sample) ---")
    if len(dataset.original_systems) > 0:
         sample_sys_idx = 0 # Look at the first loaded original system
         print(f"Sample System Label: {dataset.original_systems[sample_sys_idx]['label']}")
         # Should come from filename parsing
         print(f"Sample System dlambda (from filename): {dataset.original_systems[sample_sys_idx]['dlambda']}")
         # Should come from free_ene.csv
         print(f"Sample System Total dG (from free_ene.csv, kcal/mol): {dataset.system_target_dg[sample_sys_idx]}")
         # Should come from free_ene.csv (kcal/mol)
         print(f"Sample System Window dGs (from free_ene.csv, kcal/mol, length {len(dataset.original_systems[sample_sys_idx]['window_dGs'])}): {dataset.original_systems[sample_sys_idx]['window_dGs'][:5]}...") # Print first 5


    if len(dataset) > 0:
        # 测试获取第一个子集样本
        print("\nTesting getitem(0)...")
        # Unpack the return values from __getitem__
        processed_data_dict_100_grid, original_system_window_dGs, original_system_dlambda, target_total_dg_tensor, system_label, all_original_mu_targets_tensor, all_original_sigma_targets_tensor, all_original_100_grid_indices_tensor = dataset[0]

        print(f"Processed data shape: {processed_data_dict_100_grid['data'].shape}") # Should be [C_out, 100, processor_max_data] after squeeze
        # 验证经过处理后，数据点数仍为 max_data (50)
        assert processed_data_dict_100_grid['data'].shape[2] == processor_max_data, \
            f"Error: LambdaDataProcessor did not pad to max_data. Expected {processor_max_data}, got {processed_data_dict_100_grid['data'].shape[2]}"
        
        # 验证原始数据长度反映了截断
        # 找到一个有效窗口的原始长度
        valid_window_mask = processed_data_dict_100_grid['masks']['window'] > 0
        if valid_window_mask.any():
            first_valid_idx = torch.where(valid_window_mask)[0][0]
            actual_length_in_processor_output = processed_data_dict_100_grid['original_lengths'][first_valid_idx].item()
            print(f"Actual data length for a valid window in processor output (should be <= {test_per_lambda_max_points}): {actual_length_in_processor_output}")
            assert actual_length_in_processor_output <= test_per_lambda_max_points, \
                f"Error: Original length for valid window ({actual_length_in_processor_output}) exceeds test_per_lambda_max_points ({test_per_lambda_max_points})."


        print(f"Processed lambdas shape: {processed_data_dict_100_grid['lambdas'].shape}") # Should be [100]
        print(f"Processed masks['window'] shape: {processed_data_dict_100_grid['masks']['window'].shape}") # Should be [100]

        # print(f"Processed data: {processed_data_dict_100_grid['data']}") # Avoid printing large tensors
        # print(f"Processed lambdas: {processed_data_dict_100_grid['lambdas']}")
        # print(f"Processed deltas: {processed_data_dict_100_grid['deltas']}")
        # print(f"Processed masks['window']: {processed_data_dict_100_grid['masks']['window']}")

        print(f"Original System Window dGs shape (for sample 0, kcal/mol): {original_system_window_dGs.shape}")
        print(f"Original System dlambda (for sample 0, from filename): {original_system_dlambda}")
        print(f"Target Total dG (for sample 0, kcal/mol): {target_total_dg_tensor.item()}")
        print(f"System Label (for sample 0): {system_label}")
        # --- NEW: Print subset target shapes ---
        print(f"Subset Mu Targets shape (for sample 0): {all_original_mu_targets_tensor.shape}") # Should be [num_original_windows_total]
        print(f"Subset Sigma Targets shape (for sample 0): {all_original_sigma_targets_tensor.shape}") # Should be [num_original_windows_total]
        print(f"Sample 0 Subset Mu Targets (first 5): {all_original_mu_targets_tensor[:5].cpu().numpy()}")
        print(f"Sample 0 Subset Sigma Targets (first 5): {all_original_sigma_targets_tensor[:5].cpu().numpy()}")
        # --- END NEW ---

        # aligned_lambdas tensor 包含了填充后的 100 个 lambda 值
        aligned_lambdas = processed_data_dict_100_grid['lambdas'] # Shape [100] after squeeze(0)
        # window_mask 标记了哪些位置是原始数据填充进来的有效窗口
        window_mask = processed_data_dict_100_grid['masks']['window'] # Shape [100]
        # 可以结合 mask 来找出原始窗口对应的 lambda 值
        # Note: Since sampling is now random, these original subset lambdas will not necessarily be sorted or equally spaced.
        original_subset_lambdas_indices = torch.where(window_mask > 0)[0]
        original_subset_lambdas = aligned_lambdas[original_subset_lambdas_indices]
        print(f"Original Lambdas in this *subset* (extracted using mask, unsorted): {original_subset_lambdas.cpu().numpy()}")

        # 创建DataLoader测试批处理
        batch_size = 4  # Example batch size
        shuffle = True
        # Pass the dataset and the custom collate function to DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

        print(f"\nTesting DataLoader with batch size {batch_size}...")
        # Unpack the return values from collate_fn
        for batch_processed_data_dict, batch_original_window_dGs_list, batch_original_dlambda_list, batch_target_total_dg, batch_system_label, batch_all_original_mu_targets, batch_all_original_sigma_targets, batch_all_original_100_grid_indices in data_loader:
            print(f"Batch processed data shape: {batch_processed_data_dict['data'].shape}")  # Should be [batch_size, C_out, 100, processor_max_data]
            print(f"Batch original window dGs list length: {len(batch_original_window_dGs_list)}")
            if len(batch_original_window_dGs_list) > 0:
                 # Shape is [num_original_windows] for each sample, variable across batch
                 print(f"  Shape of first item in batch_original_window_dGs_list: {batch_original_window_dGs_list[0].shape}")
            print(f"Batch original dlambda list length: {len(batch_original_dlambda_list)}")
            if len(batch_original_dlambda_list) > 0:
                 print(f"  First item in batch_original_dlambda_list: {batch_original_dlambda_list[0]}") # List of floats
            print(f"Batch target total dG shape: {batch_target_total_dg.shape}")  # Should be [batch_size] (kcal/mol)
            print(f"Batch system label: {batch_system_label}")
            # --- NEW: Print batched subset target shapes ---
            # Note: These are lists of tensors with variable length per sample in the batch
            print(f"Batch All Original Mu Targets is a list of {len(batch_all_original_mu_targets)} tensors.")
            if batch_all_original_mu_targets:
                 print(f"  Shape of first All Original Mu Targets tensor: {batch_all_original_mu_targets[0].shape}") # [num_original_windows_total_i]
            print(f"Batch All Original Sigma Targets is a list of {len(batch_all_original_sigma_targets)} tensors.")
            if batch_all_original_sigma_targets:
                 print(f"  Shape of first All Original Sigma Targets tensor: {batch_all_original_sigma_targets[0].shape}") # [num_original_windows_total_i]
            print(f"Batch All Original 100 Grid Indices is a list of {len(batch_all_original_100_grid_indices)} tensors.")
            if batch_all_original_100_grid_indices:
                 print(f"  Shape of first All Original 100 Grid Indices tensor: {batch_all_original_100_grid_indices[0].shape}") # [num_original_windows_total_i]
            # --- END NEW ---
            break # Just process one batch

    else:
        print("\nNo subset samples generated. Check data directory and subset parameters.")

# ... original LambdaDataProcessor test code commented out or removed ...

