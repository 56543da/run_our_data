# MMCAF-Net：Adam 训练 NaN 复盘与修复报告

## 背景与现象

在 Windows 服务器（L20）上使用 Adam 训练 Exp4 时，出现两类问题：

1. **梯度非有限（Inf/NaN）**  
   日志出现 `Warning: non-finite grad_norm; skipping update`，并在增强日志后能定位到具体参数名，例如：
   - `bad_grads=['module.image_encoder.dcfb3.msdc.dwconvs.0.0.weight']`

2. **前向输出非有限（out/loss 变 NaN）**  
   日志出现 `Warning: non-finite detected; skipping backward`，并且 `out_finite=False loss=nan`。

SGD 在相同数据/结构下相对更“抗崩”，而 Adam 更容易进入持续 NaN（动量状态被污染后难以恢复）。

## 根因分析（为什么会 NaN）

### 1) 训练端早期缺少“非有限值隔离”

虽然验证器里对输入做过 `nan_to_num`/NaN 检测，但训练循环里如果：
- 输入包含 NaN/Inf
- 或某层数值溢出导致 `out/loss` 非有限

则会在 backward 阶段产生非有限梯度；对 Adam 来说，一旦 `optimizer.step()` 发生在梯度非有限时，会污染 `exp_avg/exp_avg_sq`，后续更容易持续 NaN。

相关实现位置：训练主循环 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

### 2) 实际触发点：图像编码器在混合精度下的梯度 NaN

在增强日志中，出现：
- `bad_grads=['module.image_encoder.dcfb3.msdc.dwconvs.0.0.weight']`
- 对应 `bad_grad_stats` 显示该梯度为 NaN（dtype 为 fp32，但 NaN 来自反传数值本身）

同时该批次：
- `img_min/img_max` 正常
- `tab_min/tab_max` 正常
- `loss/out` 有限

这说明问题不是数据直接带 NaN/Inf，也不是 KAN 分支先炸；而是 **image_encoder 的某个 depthwise conv 分支在 AMP（尤其 fp16 autocast + scaler）下发生数值不稳定，导致反向传播梯度 NaN**。

## 修复方案与原因（为什么现在正常）

### A. 输入端清洗（防止 NaN/Inf 直接传播）

对 `img/tab/label` 做 `isfinite` 检测并 `nan_to_num` 清洗，保证进入网络的张量是有限值。

相关实现位置：训练循环中输入搬运后 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

### B. 前向后立即隔离非有限 out/loss（阻断污染）

在 backward 前检查：
- `out` 是否全部有限
- `loss` 是否有限

只要出现非有限，直接跳过该 batch 的 backward，避免梯度与 Adam 状态污染。

相关实现位置：前向结束后 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

### C. 反向后检测梯度（以 grad_norm 为闸门）

在 `unscale_(optimizer)` 后执行：
- `clip_grad_norm_`
- 如果 `grad_norm` 非有限：跳过 `optimizer.step()` 并更新 scaler（fp16 情况）

同时打印：
- 当前学习率、loss、out/img/tab 的范围
- 最多 5 个非有限梯度的参数名与梯度 min/max/dtype

这样能在服务器上直接定位“是哪一层开始炸”。

相关实现位置：梯度裁剪与日志 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

### D. 对不稳定模块局部关闭 autocast（关键修复）

根据 `bad_grads` 定位到 image_encoder 的 depthwise conv，采取与 KAN 分支一致的策略：

- **KAN 分支**：在 autocast 开启时强制用 fp32 计算（降低 B-spline/缩放的数值风险）  
  位置：[MMCAF_Net.py](file:///e:/run_our_data/MMCAF-Net-main/models/MMCAF_Net.py)

- **image_encoder 分支**：在 autocast 开启时强制用 fp32 计算（抑制卷积反传 NaN）  
  位置：[MMCAF_Net.py](file:///e:/run_our_data/MMCAF-Net-main/models/MMCAF_Net.py)

这一步是“现在训练恢复正常”的主要原因：它针对的是日志明确命中的炸点层，而不是泛泛地降低学习率或盲目改超参。

### E. 更稳的 AMP 模式选择（优先 bf16）

在支持 bf16 的 GPU 上（如 L20），优先使用 `bfloat16` autocast：
- bf16 具有更大的指数范围，通常比 fp16 更不容易溢出产生 Inf/NaN
- bf16 通常不需要 GradScaler

不支持 bf16 时才退回 fp16 + GradScaler，并把 init_scale 调低以减少溢出概率。

相关实现位置：AMP 初始化 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

## “为什么现在日志看起来重复很多”

你看到的类似：

- `Epoch 1: 98%|...| 179/182 ...`
- 中间夹着一行 `[epoch: 1, iter: ...]`
- 然后又出现同一条 `Epoch 1: 98%|...| 179/182 ...`

主要原因是：
- tqdm 进度条会反复刷新当前行（回车覆盖），并且当中间出现 `print/tqdm.write` 时，tqdm 会重新渲染进度条，视觉上像“重复打印”。

已做的输出优化：
- logger 输出改用 `tqdm.write`（避免与进度条抢 stdout）  
  位置：[base_logger.py](file:///e:/run_our_data/MMCAF-Net-main/logger/base_logger.py)
- tqdm 进度条降低刷新频率，并把 postfix 更新改为每 10 个 batch 更新一次  
  位置：[train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)

## 结论

本次 Adam NaN 的核心原因是 **图像编码器分支在混合精度下出现梯度 NaN（定位到具体 depthwise conv 权重）**，并且训练循环缺少“非有限值隔离”会让 Adam 状态更易被污染。

通过：
- 输入清洗
- 前向后 out/loss finite 检测
- 反向后 grad_norm finite 检测 + 定位坏梯度参数
- 对 image_encoder/KAN 局部关闭 autocast（fp32）
- 优先 bf16 autocast（若支持）

训练已恢复稳定，不再出现持续 NaN。

