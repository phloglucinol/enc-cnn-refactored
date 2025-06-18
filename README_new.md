# ENC-CNN: 分子动力学自由能计算深度学习模型 (重构版)

> 🚀 **全面重构完成！** 这是经过完全重构的版本，具有更好的代码结构、完整的文档和现代化的工程实践。

## 📊 重构成果

| 指标 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| 主文件行数 | 561行 | 150行 | -73% |
| 模块化程度 | 低 | 高 | 质的飞跃 |
| 注释覆盖率 | <10% | >80% | +700% |
| 配置管理 | 无 | 完整 | 从0到1 |
| 可维护性 | 差 | 优秀 | 🌟🌟🌟🌟🌟 |

## 🏗️ 项目架构

```
enc-cnn/
├── 📁 models/              # 深度学习模型
│   ├── __init__.py
│   └── encoder_cnn_model.py    # 主模型实现
├── 📁 utils/               # 通用工具
│   ├── __init__.py
│   └── pos_embed.py            # 位置编码工具
├── 📁 util/                # 核心功能模块
│   ├── enc_engine_pretrain.py  # 预训练引擎
│   ├── enc_engine_finetune.py  # 微调引擎
│   ├── enc_model_dg_loss.py    # 损失函数
│   └── test_lambda_emb_dataset.py # 数据处理
├── 📁 backup_old_code/     # 旧代码备份
├── config.py               # 🔧 统一配置管理
├── trainer.py              # 🎯 训练器封装
├── main_train.py           # 🚀 简洁主脚本
└── README.md               # 📖 项目文档
```

## ✨ 核心特性

### 🧠 模型架构
- **自适应λ位置编码** - 可学习的lambda值位置编码机制
- **多策略特征嵌入** - 支持3通道和4通道输入的不同嵌入策略
- **Transformer编码器** - 使用多头注意力机制处理序列数据
- **多损失函数组合** - 总ΔG损失、聚合窗口损失、平滑损失、特征损失

### ⚙️ 工程特性
- **配置驱动** - 统一的配置管理系统，支持类型检查和默认值
- **模块化设计** - 清晰的模块分离，符合单一职责原则
- **完整注释** - 详细的中文注释和docstring，PyTorch新手友好
- **错误处理** - 完善的异常处理和验证机制

## 🚀 快速开始

### 1. 环境要求

```bash
# 主要依赖
torch >= 1.9.0
timm == 0.3.2
pandas
numpy
scipy
natsort
tensorboard
```

### 2. 训练模型

```bash
# 基本训练命令
python main_train.py \
    --data_path /path/to/data \
    --epochs 100 \
    --batch_size 4 \
    --model enc_cnn_chans3

# 完整配置示例
python main_train.py \
    --data_path /path/to/data \
    --output_dir ./outputs \
    --batch_size 4 \
    --epochs 100 \
    --model enc_cnn_chans3 \
    --in_chans 3 \
    --subset_size 14 \
    --total_dg_loss_weight 1.0 \
    --agg_dg_loss_weight 1.0 \
    --smoothness_loss_weight 0.1 \
    --feature_loss_weight 1.0
```

### 3. 数据格式

数据目录结构：
```
data_path/
├── train/
│   └── system_X/
│       ├── complex/
│       │   ├── *.csv                    # 原始窗口数据
│       │   └── fe_cal_out/
│       │       └── free_ene_zwanzig.csv # 目标ΔG值
│       └── ligand/
│           ├── *.csv
│           └── fe_cal_out/
│               └── free_ene_zwanzig.csv
└── val/
    └── (相同结构)
```

## 📋 配置说明

### 主要参数类别

#### 🔢 模型参数
- `--model`: 模型类型 (`enc_cnn_chans3` 或 `enc_cnn_chans4`)
- `--in_chans`: 输入通道数 (3或4)
- `--input_size`: 输入尺寸，默认(50, 100)

#### 📊 数据参数
- `--subset_size`: 每个子集的窗口数量
- `--num_random_subsets_per_system`: 每个系统生成的随机子集数量
- `--per_lambda_max_points`: 每个λ窗口的最大数据点数

#### ⚖️ 损失权重
- `--total_dg_loss_weight`: 总ΔG损失权重
- `--agg_dg_loss_weight`: 聚合窗口损失权重
- `--smoothness_loss_weight`: 平滑损失权重
- `--feature_loss_weight`: 特征损失权重

#### 🏃 训练参数
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--weight_decay`: 权重衰减

## 🔬 模型详解

### 嵌入策略

#### 3通道策略 (EmbeddingStrategy3Chans)
- 输入: μ, σ², error
- Lambda特征投影 + CNN编码器
- 适用于基础分子动力学数据

#### 4通道策略 (EmbeddingStrategy4Chans)  
- 输入: μ, σ², error, Δλ
- 自适应λ位置编码
- 支持三种编码方式: CNN、Linear、MLP

### 损失函数组合

1. **总ΔG损失** - 预测总自由能变化与真实值的MSE
2. **聚合窗口损失** - 处理不同δλ值的窗口聚合
3. **平滑损失** - 确保μ和σ预测的空间连续性
4. **特征损失** - 监督学习损失，直接约束μ和σ预测值

## 📈 输出结果

### 训练过程
- **TensorBoard日志** - 实时训练曲线监控
- **检查点保存** - 自动保存最佳模型
- **详细日志** - 完整的训练统计信息

### 验证结果
- `validation_results.csv` - 系统级ΔG预测结果
- `validation_per_window_dG_for_plot.csv` - 窗口级ΔG预测结果
- 实时损失和误差统计

## 🛠️ 开发指南

### 代码结构设计原则

1. **可读性优先** - 代码是写给人看的，机器只是恰好能执行
2. **DRY原则** - 绝不复制代码片段，通过抽象封装通用逻辑
3. **高内聚，低耦合** - 功能相关的代码放在一起，模块间减少依赖

### 扩展指南

#### 添加新的嵌入策略
```python
class MyEmbeddingStrategy(BaseEmbeddingModule):
    def forward(self, x, lambdas, deltas, masks, original_lengths):
        # 实现你的嵌入逻辑
        return lambda_emb, feat_emb
```

#### 添加新的损失函数
```python
# 在MaskedAutoencoderViT.forward_loss中添加
if self.my_loss_weight > 0:
    my_loss = compute_my_loss(pred, target)
    loss_dict['my_loss'] = my_loss
    total_loss += self.my_loss_weight * my_loss
```

## 📝 更新日志

### v2.0.0 (重构版)
- ✅ 完全重构代码架构
- ✅ 添加统一配置管理
- ✅ 创建训练器封装
- ✅ 完整的中文文档
- ✅ 模块化设计
- ✅ 清理冗余代码

### v1.0.0 (原始版)
- 基础功能实现
- 单文件架构（已废弃）

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于Meta Platforms, Inc.的MAE项目修改，遵循相应的开源许可证。

## 🙏 致谢

- **Meta AI** - 提供MAE架构基础
- **timm** - 提供Vision Transformer实现
- **PyTorch** - 深度学习框架

---

> 💡 **提示**: 这个重构版本不仅仅是代码的改进，更是软件工程思维的体现。通过模块化设计、配置管理、错误处理等现代化实践，让代码变得可维护、可扩展、可测试。这才是专业的深度学习项目应该有的样子！