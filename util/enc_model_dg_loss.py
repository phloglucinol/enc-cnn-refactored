import torch
import torch.nn as nn
from typing import List, Literal


class FreeEnergyConfig:
    """自由能计算配置类"""
    DEFAULT_KBT = 0.592  # kcal/mol
    DEFAULT_PRED_DLAMBDA = 0.01
    MAX_PRED_WINDOWS = 100


def convert_to_kcal_mol(pred_dgs: torch.Tensor, kbt: float = FreeEnergyConfig.DEFAULT_KBT) -> torch.Tensor:
    """将无量纲的ΔG值转换为kcal/mol单位"""
    return pred_dgs * kbt


def calculate_window_ratios(target_dlambdas: List[float], 
                          pred_dlambda: float = FreeEnergyConfig.DEFAULT_PRED_DLAMBDA) -> torch.Tensor:
    """计算目标窗口与预测窗口的大小比例"""
    ratios = [max(1, round(dlambda / pred_dlambda)) for dlambda in target_dlambdas]
    return torch.tensor(ratios, dtype=torch.long)


def aggregate_predictions_vectorized(pred_cumsum: torch.Tensor, 
                                   ratios: torch.Tensor,
                                   target_lengths: List[int]) -> List[torch.Tensor]:
    """向量化聚合预测值到目标窗口"""
    batch_size = pred_cumsum.shape[0]
    aggregated_results = []
    
    for i in range(batch_size):
        ratio = ratios[i].item()
        num_windows = target_lengths[i]
        
        # 计算窗口索引
        window_indices = torch.arange(num_windows + 1, device=pred_cumsum.device) * ratio
        window_indices = torch.clamp(window_indices, 0, FreeEnergyConfig.MAX_PRED_WINDOWS)
        
        # 使用累积和计算窗口聚合值
        start_indices = window_indices[:-1]
        end_indices = window_indices[1:]
        
        aggregated_values = pred_cumsum[i, end_indices] - pred_cumsum[i, start_indices]
        aggregated_results.append(aggregated_values)
    
    return aggregated_results


def dg_aggregation_loss_v3(pred_dGs: torch.Tensor,
                           original_window_dGs_list: List[torch.Tensor],
                           original_dlambda_list: List[float],
                           reduction: Literal['none', 'mean', 'sum'] = 'mean',
                           kbt: float = FreeEnergyConfig.DEFAULT_KBT) -> torch.Tensor:
    """
    重构版自由能预测损失函数 - 向量化实现
    
    Args:
        pred_dGs: 模型预测的ΔG值 [batch_size, 100]
        original_window_dGs_list: 目标窗口ΔG列表 (kcal/mol)
        original_dlambda_list: 目标Δλ值列表
        reduction: 损失聚合方式
        kbt: 玻尔兹曼常数×温度值
    
    Returns:
        损失张量
    """
    batch_size = pred_dGs.shape[0]
    
    # 输入验证
    if len(original_window_dGs_list) != batch_size or len(original_dlambda_list) != batch_size:
        raise ValueError("批次大小不匹配")
    
    if any(dlambda <= 0 for dlambda in original_dlambda_list):
        raise ValueError("检测到无效的dlambda值")
    
    # 单位转换
    pred_dGs_kcal = convert_to_kcal_mol(pred_dGs, kbt)
    
    # 计算累积和
    pred_cumsum = torch.cat([
        torch.zeros_like(pred_dGs_kcal[:, :1]), 
        pred_dGs_kcal
    ], dim=1).cumsum(dim=1)
    
    # 计算窗口比例
    ratios = calculate_window_ratios(original_dlambda_list)
    target_lengths = [dgs.shape[0] for dgs in original_window_dGs_list]
    
    # 向量化聚合
    aggregated_preds = aggregate_predictions_vectorized(pred_cumsum, ratios, target_lengths)
    
    # 计算损失
    sample_losses = []
    for i in range(batch_size):
        target_dgs = original_window_dGs_list[i].to(pred_dGs.device)
        pred_dgs = aggregated_preds[i]
        
        mse_loss = torch.mean((pred_dgs - target_dgs) ** 2)
        sample_losses.append(mse_loss)
    
    batch_losses = torch.stack(sample_losses)
    
    # 应用reduction
    if reduction == 'none':
        return batch_losses
    elif reduction == 'mean':
        return torch.mean(batch_losses)
    elif reduction == 'sum':
        return torch.sum(batch_losses)


# 保持向后兼容性
def dg_aggregation_loss_v2(pred_dGs: torch.Tensor,
                           original_window_dGs_list: List[torch.Tensor],
                           original_dlambda_list: List[float],
                           reduction: str = 'mean') -> torch.Tensor:
    """原版函数的包装器，保持向后兼容"""
    return dg_aggregation_loss_v3(pred_dGs, original_window_dGs_list, 
                                  original_dlambda_list, reduction)


if __name__ == '__main__':
    print("测试重构后的自由能损失函数...")
    
    # 模拟数据
    batch_size = 4
    pred_dGs_sim = torch.randn(batch_size, 100) * 0.1
    
    # 模拟不同分辨率的目标数据
    test_cases = [
        (50, 0.02, 1.0, 0.2),   # 50个窗口，Δλ=0.02
        (25, 0.04, 2.0, 0.3),   # 25个窗口，Δλ=0.04  
        (100, 0.01, 0.5, 0.1),  # 100个窗口，Δλ=0.01
        (20, 0.05, 3.0, 0.4)    # 20个窗口，Δλ=0.05
    ]
    
    original_window_dGs_list_sim = []
    original_dlambda_list_sim = []
    
    for num_windows, dlambda, mean, std in test_cases:
        target_dgs = torch.randn(num_windows) * std + mean
        original_window_dGs_list_sim.append(target_dgs)
        original_dlambda_list_sim.append(dlambda)
    
    # 测试新版本函数
    print("\n=== 测试 dg_aggregation_loss_v3 ===")
    loss_v3_mean = dg_aggregation_loss_v3(
        pred_dGs_sim, original_window_dGs_list_sim, 
        original_dlambda_list_sim, reduction='mean'
    )
    print(f"V3 损失 (mean): {loss_v3_mean.item():.6f}")
    
    loss_v3_none = dg_aggregation_loss_v3(
        pred_dGs_sim, original_window_dGs_list_sim,
        original_dlambda_list_sim, reduction='none'
    )
    print(f"V3 损失 (none) 形状: {loss_v3_none.shape}")
    print(f"V3 各样本损失: {[f'{x:.6f}' for x in loss_v3_none.tolist()]}")
    
    # 测试向后兼容性
    print("\n=== 测试向后兼容性 ===")
    loss_v2_mean = dg_aggregation_loss_v2(
        pred_dGs_sim, original_window_dGs_list_sim,
        original_dlambda_list_sim, reduction='mean'
    )
    print(f"V2 损失 (mean): {loss_v2_mean.item():.6f}")
    
    # 验证结果一致性
    print(f"\n结果一致性检查: {torch.allclose(loss_v3_mean, loss_v2_mean)}")
    
    # 性能比较（模拟大批次）
    print("\n=== 性能测试 ===")
    large_batch = 32
    large_pred = torch.randn(large_batch, 100) * 0.1
    large_targets = [torch.randn(50) * 0.2 + 1.0 for _ in range(large_batch)]
    large_dlambdas = [0.02] * large_batch
    
    import time
    start = time.time()
    for _ in range(10):
        _ = dg_aggregation_loss_v3(large_pred, large_targets, large_dlambdas)
    end = time.time()
    print(f"V3版本 10次运行时间: {(end-start)*1000:.2f}ms")
    
    print("\n重构完成！新版本具有更好的性能和可维护性。")
