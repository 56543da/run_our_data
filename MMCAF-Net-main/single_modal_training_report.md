# 单模态分支训练策略报告

## 1. 概述
本报告描述了 MMCAF-Net 在单模态（Image-Only 和 Tabular-Only）训练模式下的模块冻结与训练策略。该策略旨在确保在仅使用一种模态数据时，能够正确微调相关特征提取器、融合模块及分类头，同时冻结缺失模态的模块以保持稳定性。

## 2. 训练模式详解

### 2.1 图像单模态 (Image-Only Mode)
- **参数配置**: `--train_mode=img_only`
- **输入数据**: 仅使用 CT 图像数据，表格数据被屏蔽（置零或忽略）。
- **模块状态**:
  - **Image Encoder (图像编码器)**: ✅ **训练中** (Unfrozen)
  - **Tabular Encoder (KAN)**: ❄️ **冻结** (Frozen, Eval Mode)
  - **Multiscale Fusion (多尺度融合)**: ✅ **训练中** (Unfrozen) - *允许特征在融合层适应*
  - **Classification Head (分类头)**: ✅ **训练中** (Unfrozen)
- **前向传播**:
  - 调用 `model.forward(..., mode=1)`。
  - 强制跳过 KAN 编码器的计算，或将其输出置零，避免噪声干扰。

### 2.2 表格单模态 (Tabular-Only Mode)
- **参数配置**: `--train_mode=tab_only`
- **输入数据**: 仅使用临床表格数据，图像数据被屏蔽。
- **模块状态**:
  - **Image Encoder (图像编码器)**: ❄️ **冻结** (Frozen, Eval Mode)
  - **Tabular Encoder (KAN)**: ✅ **训练中** (Unfrozen)
  - **Multiscale Fusion (多尺度融合)**: ✅ **训练中** (Unfrozen) - *允许特征在融合层适应*
  - **Classification Head (分类头)**: ✅ **训练中** (Unfrozen)
- **前向传播**:
  - 调用 `model.forward(..., mode=2)`。
  - 强制跳过 Image Encoder 的计算，直接使用表格特征进入融合层。

## 3. 学习率与优化器配置
- **独立学习率**: 
  - 在 `run_multi_experiments.ps1` 中，可为 Image-Only 和 Tabular-Only 实验分别设置不同的 `lr`。
  - 推荐：图像分支通常需要较小学习率（如 1e-4），表格分支（KAN）可能需要较大学习率（如 1e-3 或 1e-2）。
- **断点续训 (Resume)**:
  - **仅恢复权重 (`resume=$true`, `resume_optim=$false`)**: 推荐模式。使用新的学习率重新初始化优化器。
  - **完全恢复 (`resume_optim=$true`)**: 恢复之前的优化器状态（包括旧的学习率）。**注意**：如需修改学习率，必须使用 `--override_lr` 参数。

## 4. 实验配置示例 (PowerShell)
在 `run_multi_experiments.ps1` 中配置：

```powershell
$experiments = @(
    # 图像单模态实验
    @{ name = "ImgOnly_Exp"; lr = "1e-4"; train_mode = "img_only"; ... },
    
    # 表格单模态实验
    @{ name = "TabOnly_Exp"; lr = "1e-2"; train_mode = "tab_only"; ... }
)
```

## 5. TensorBoard 指标
- Ablation Study 表格与曲线在单模态训练中依然保留，但会针对单模态进行适配（如只显示相关模态的贡献）。
- 验证集评估会自动根据训练模式调整。
