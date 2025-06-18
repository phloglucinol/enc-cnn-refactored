# ENC-CNN: 基于编码器-CNN的分子自由能计算模型

本项目实现了一个基于Transformer编码器和CNN的深度学习模型，用于分子动力学模拟中的自由能差计算。该模型结合了Lambda窗口采样技术和多种损失函数，能够预测分子系统的热力学性质。

## 项目概述

该模型是基于Meta's MAE (Masked Autoencoder)架构改进的深度学习框架，专门用于处理分子动力学模拟数据中的λ窗口采样数据，预测分子系统的自由能变化(ΔG)。

### 核心特性

- **多通道输入支持**: 支持3通道(μ, σ, error)和4通道(μ, σ, error, δλ)输入
- **自适应λ位置编码**: 可学习的λ值位置编码机制
- **多种嵌入策略**: CNN、扁平化线性层、MLP等多种特征嵌入方法
- **复合损失函数**: 包含总ΔG损失、聚合窗口损失、平滑损失和特征损失
- **灵活的数据处理**: 支持可变长度窗口和随机子集采样

## 文件结构

```
enc-cnn/
├── README.md                          # 项目说明文档
├── CLAUDE.md                          # Claude助手配置文件
├── enc+cnn_main_train+val.py          # 主训练脚本
├── encoder_cnn_model_test_v1.py       # 核心模型定义
└── util/                              # 工具模块
    ├── enc_engine_finetune.py         # 验证引擎
    ├── enc_engine_pretrain.py         # 训练引擎
    ├── enc_model_dg_loss.py           # ΔG聚合损失计算
    └── test_lambda_emb_dataset.py     # 数据集和数据处理器
```

## 核心组件

### 1. 模型架构 (`encoder_cnn_model_test_v1.py`)

#### MaskedAutoencoderViT
主要的模型类，包含以下关键组件：

- **嵌入模块选择**: 根据输入通道数(3或4)选择合适的嵌入策略
- **Transformer编码器**: 使用多头注意力机制处理序列数据
- **投影头**: 将编码器输出映射为3个特征(μ, σ, error)

#### 嵌入策略

1. **EmbeddingStrategy3Chans**: 
   - 适用于3通道输入
   - 使用Lambda特征投影和CNN编码器
   - 输入: (μ, σ, error)

2. **EmbeddingStrategy4Chans**:
   - 适用于4通道输入
   - 使用自适应λ位置编码
   - 支持三种特征编码方式: CNN、扁平化线性层、MLP
   - 输入: (μ, σ, error, δλ)

#### 自适应λ编码 (`AdaptiveLambdaEncoding`)
- 可学习的缩放因子C
- 基于正弦-余弦位置编码
- 为不同λ值提供可区分的位置信息

### 2. 数据处理 (`test_lambda_emb_dataset.py`)

#### LambdaDataProcessor
负责将原始λ窗口数据标准化为100个窗口的固定网格：

- **标准化网格**: 将数据对齐到λ∈[0.00, 0.01, ..., 0.99]的100窗口网格
- **掩码机制**: 标记有效窗口和填充窗口
- **长度记录**: 保存每个窗口的原始数据点数量

#### CustomDataset
数据集类，负责：

- **系统加载**: 从CSV文件加载分子系统数据
- **目标提取**: 从free_ene.csv读取真实ΔG值
- **随机采样**: 为每个系统生成多个随机子集
- **批处理**: 通过custom_collate_fn支持可变长度数据的批处理

### 3. 损失函数

#### 复合损失函数
模型使用四种损失的加权组合：

1. **总ΔG损失** (`total_dg_loss_weight`):
   - MSE损失，比较预测总ΔG与真实值
   - 公式: `(predicted_total_dg - target_total_dg)²`

2. **聚合窗口损失** (`agg_dg_loss_weight`):
   - 处理不同δλ值的窗口聚合
   - 通过`dg_aggregation_loss_v2`实现
   - 将100个预测窗口聚合到原始窗口数量

3. **平滑损失** (`smoothness_loss_weight`):
   - 确保μ和σ预测的空间连续性
   - 基于二阶导数的平滑度约束
   - 公式: `mean((f[i+2] - 2*f[i+1] + f[i])²)`

4. **特征损失** (`feature_loss_weight`):
   - 监督学习损失，直接约束μ和σ预测值
   - 使用原始窗口的真实μ、σ值作为监督信号

### 4. 训练和验证

#### 训练引擎 (`enc_engine_pretrain.py`)
- 支持梯度累积和学习率调度
- 实时监控各项损失指标
- 保存训练结果到CSV文件

#### 验证引擎 (`enc_engine_finetune.py`)
- 无梯度验证过程
- 计算平均绝对误差(MAE)
- 生成预测vs真实值的对比报告
- 保存每窗口ΔG结果用于可视化

## 使用方法

### 1. 环境要求

```bash
# 主要依赖
torch >= 1.9.0
timm == 0.3.2
pandas
numpy
scipy
natsort
```

### 2. 数据准备

数据目录结构应为：
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

CSV文件命名格式: `*_lambda{λ值}_delta{δλ值}.csv`

### 3. 训练模型

```bash
python enc+cnn_main_train+val.py \
    --data_path /path/to/data \
    --output_dir ./output \
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

### 4. 主要参数

#### 模型参数
- `--model`: 模型类型 (`enc_cnn_chans3` 或 `enc_cnn_chans4`)
- `--in_chans`: 输入通道数 (3或4)
- `--input_size`: 输入尺寸，默认(50, 100)

#### 数据参数
- `--subset_size`: 每个子集的窗口数量
- `--num_random_subsets_per_system`: 每个系统生成的随机子集数量
- `--per_lambda_max_points`: 每个λ窗口的最大数据点数
- `--processor_include_delta`: 是否在数据中包含δλ通道

#### 损失权重
- `--total_dg_loss_weight`: 总ΔG损失权重
- `--agg_dg_loss_weight`: 聚合窗口损失权重
- `--smoothness_loss_weight`: 平滑损失权重
- `--feature_loss_weight`: 特征损失权重

## 输出结果

### 训练过程
- **TensorBoard日志**: 在`--log_dir`中保存训练曲线
- **检查点**: 在`--output_dir`中保存最佳模型
- **日志文件**: 详细的训练统计信息

### 验证结果
- **validation_results.csv**: 系统级ΔG预测结果
- **validation_per_window_dG_for_plot.csv**: 窗口级ΔG预测结果
- **控制台输出**: 实时损失和误差统计

## 模型详细设计

### 数据流程

1. **数据加载**: CustomDataset从CSV文件读取原始窗口数据
2. **预处理**: LambdaDataProcessor将数据标准化到100窗口网格
3. **统计计算**: 计算训练集的μ、σ、error通道的均值和标准差
4. **归一化**: 在forward_encoder中对前3个通道进行Z-score归一化
5. **嵌入**: 根据通道数选择合适的嵌入策略
6. **编码**: Transformer编码器处理序列特征
7. **预测**: 投影头输出3个特征的预测值
8. **反归一化**: 将归一化的预测值转换回原始尺度
9. **损失计算**: 计算复合损失函数

### 关键算法

#### ΔG计算公式
```
dG_n = μ - σ²/2 + error  (每窗口)
总ΔG = Σ(dG_n * kbt)     (kbt = 0.592)
```

#### 聚合损失算法
1. 将100个预测窗口按比例聚合到原始窗口数量
2. 比例 = 原始δλ / 0.01
3. 使用累积和进行高效聚合
4. 计算聚合预测与真实值的MSE

## 许可证

本项目基于Meta Platforms, Inc.的MAE项目修改，遵循相应的开源许可证。

## 参考文献

- **DeiT**: https://github.com/facebookresearch/deit
- **BEiT**: https://github.com/microsoft/unilm/tree/master/beit
- **MAE**: Meta's Masked Autoencoder Vision Transformer

## 联系方式

如有问题或建议，请通过项目Issue页面联系。