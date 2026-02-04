# 修复SGD不平衡、清理日志及计算基准

基于您的反馈，这是更新后的计划。我将跳过此刻计算空准确率的步骤，正如您建议的那样，可以在训练开始时进行。我将专注于TensorBoard日志的清理和文档工作。

### 1. 清理 TensorBoard 日志
我将修改 `model_evaluator1.py` 以移除不想记录到 TensorBoard 的指标和消融研究。
*   **移除**: `val/Ablation/TabOnly/loss`, `val/Ablation/Multimodal/loss`, `val/Ablation/ImgOnly/AUROC`.
*   **移除**: 在数据被屏蔽的消融模式下的空指标（如 `Specificity`）。
*   **操作**: 注释掉或删除 `_eval_phase` 中的相关日志代码行。

### 2. 类别加权 vs 过采样 (ROS) 分析
您提到已经对数据集应用了 **过采样** (ROS)。
*   **分析**: 在 `preprocess_data.py` 中，随机过采样 (ROS) 确实**仅应用于训练集** (`parser='train'`).
    ```python
    # 5. Oversampling (ROS) - ONLY for Train
    if len(train_1) < target_count:
        train_1_over = train_1.sample(n=target_count, replace=True)
    ```
*   **结论**:
    *   **训练**: 训练集是平衡的 (1:1 比例)，所以模型在训练期间*应该*平等地学习两个类别。
    *   **验证**: 验证集保持 **不平衡** (原始分布)。
    *   **问题**: 即使训练是平衡的，如果验证集严重不平衡（例如 90% 负样本），一个针对“准确率”优化的模型如果对正样本特征不敏感，在验证数据上仍可能偏向于高特异性。
    *   **建议**: 既然您已经有了 ROS，向损失函数添加类别权重可能对训练阶段是 **多余甚至有害的** (双重加权)。**然而**，验证集上的高特异性/低敏感度表明模型仍然保守。
    *   **修正策略**: 我将 **不会** 向训练损失添加类别权重（以避免与 ROS 双重修正）。相反，我将在文档中强调 **阈值调整** 作为修复验证指标的主要方法，正如您建议的（“查看 ROC 曲线”）。

### 3. 文档
我将更新 `Hyperparameter_Guide.md` (超参数指南) 以反映 ROS 已经到位的情况，并解释为什么在这种情况下阈值调整是验证评估的正确方法。

---

**执行步骤:**
1.  修改 `model_evaluator1.py` 以清理日志。
2.  更新 `Hyperparameter_Guide.md` 以包含您的 ROS 实现的具体背景。
