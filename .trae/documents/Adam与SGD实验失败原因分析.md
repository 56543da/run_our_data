# Adam vs SGD: 实验失败原因深度分析

本文档旨在分析为何在相同的代码库中，`Exp1_adam_B32` (Adam) 可以正常训练，而 `Exp2_sgd_B32` (SGD) 却出现 loss 不下降的情况。

## 1. 核心原因：参数实现差异

问题的根本原因在于 `util/optim_util.py` 中对优化器的初始化逻辑，以及不同优化器对参数的敏感度差异。

### 代码实现 (util/optim_util.py)

```python
if args.optimizer == 'sgd':
    optimizer = optim.SGD(parameters, args.learning_rate,
                      momentum=args.sgd_momentum,
                      weight_decay=args.weight_decay,
                      dampening=args.sgd_dampening) # 关键点：SGD 使用了 dampening
elif args.optimizer == 'adam':
    optimizer = optim.Adam(parameters, args.learning_rate,
                           betas=(args.adam_beta_1, args.adam_beta_2), 
                           weight_decay=args.weight_decay)
    # 关键点：Adam 忽略了 dampening 参数
```

## 2. 为什么 Adam 可以工作？

在 `Exp1` 配置中：
- **Optimizer**: `adam`
- **Arguments**: 虽然脚本通过 `trainArgs` 传入了 `--sgd_dampening=0.9`，但如上代码所示，`optim.Adam` **根本不接收也不使用** 这个参数。
- **Result**: Adam 使用了默认的自适应学习率机制，忽略了不合理的阻尼设置，因此可以正常工作。

## 3. 为什么 SGD 失败？

在 `Exp2` 配置中：
- **Optimizer**: `sgd`
- **Arguments**: 脚本传入了 `--sgd_dampening=0.9`，且 `optim.SGD` **使用** 了这个参数。

### 致命组合：Dampening 0.9 + Momentum 0.9

PyTorch 中 SGD 的动量更新公式如下（简化版）：

$$v_{t+1} = \mu \cdot v_t + (1 - \tau) \cdot g_{t+1}$$

其中：
- $v$ 是速度 (velocity/momentum buffer)
- $\mu$ 是动量因子 (`momentum=0.9`)
- $\tau$ 是阻尼因子 (`dampening=0.9`)
- $g$ 是当前梯度

**后果**：
当 `dampening=0.9` 时，项 $(1 - \tau)$ 变为 $0.1$。这意味着**只有 10% 的新梯度信息**被加入到更新中。模型几乎接收不到新的学习信号，导致权重更新极度缓慢，Loss 几乎不动。

通常情况下，SGD 的 `dampening` 应该设为 **0**。

### 次要原因：学习率与权重衰减

- **学习率 (LR)**: 您原始设置的 `1e-4` 对于 SGD 来说太小了。Adam 通常适合 `1e-4` 到 `1e-5`，但 SGD 通常需要 `1e-2` 到 `1e-1` 才能有效收敛。
- **权重衰减 (Weight Decay)**: 原始设置的 `1e-2` 对于 SGD 来说是非常强的正则化（通常用 `1e-4`）。配合极小的 LR 和极大的 Dampening，模型参数被强力压向 0，而无法向 Loss 最小值的方向移动。

## 4. 解决方案总结

我们只对 `Exp2` (SGD) 做了必要的参数修正，保留了代码结构不变：

| 参数 | 原始值 (失败) | 修正值 (推荐) | 原因 |
| :--- | :--- | :--- | :--- |
| **sgd_dampening** | 0.9 | **0** | **最关键修复**。恢复正常的梯度更新权重。 |
| **learning_rate** | 1e-4 | **1e-2** | SGD 需要更大的步长来跳出局部极小值。 |
| **weight_decay** | 1e-2 | **1e-4** | 防止正则化过强导致欠拟合。 |

通过这些修改，SGD 就能像 Adam 一样正常工作了。
