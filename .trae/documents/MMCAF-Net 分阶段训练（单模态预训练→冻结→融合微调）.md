## 目标
把当前工程的训练流程扩展成你说的 3 段式：
1) 仅预训练图像编码器；2) 冻结图像编码器，仅预训练表格编码器；3) 冻结双编码器，仅训练融合模块 + 最终分类器。

## 现状梳理（关键结构）
- 多模态模型是 [MMCAF_Net.py](file:///e:/run_our_data/MMCAF-Net-main/models/MMCAF_Net.py)：
  - 图像编码器：`self.image_encoder = Img_new()`（[img_encoder1.py](file:///e:/run_our_data/MMCAF-Net-main/models/img_encoder1.py)）
  - 表格编码器：`self.kan = KANLayer(...)`（输出 24 维）
  - 融合模块：`self.multiscale_fusion`（MSCA）
  - 最终分类器：`self.fin_cls`（24→1）
- 目前训练脚本 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py) 总是 `out = model.forward(img, tab)`，优化器默认吃 `model.parameters()`，没有系统性的 freeze/unfreeze。
- 评估器 [model_evaluator1.py](file:///e:/run_our_data/MMCAF-Net-main/evaluator/model_evaluator1.py) 已经支持 `mode`（0=多模态，1=ImgOnly mask tab，2=TabOnly mask img），但它只是在“输入侧做 mask”，并不等同于“只训练某个编码器”。

## 设计方案（最小侵入、兼容现有调用）
### 1) 给 MMCAF_Net.forward 增加“输出头 head”参数
在 [MMCAF_Net.forward](file:///e:/run_our_data/MMCAF-Net-main/models/MMCAF_Net.py#L72-L109) 增加参数 `head`（默认 `"fusion"`，保证原来的 `model(img, tab)`/`model.forward(img, tab, mode=...)` 不受影响）：
- `head='img'`：只跑 `image_encoder`，直接返回 `Img_new` 的 `pred`（[img_encoder1.py](file:///e:/run_our_data/MMCAF-Net-main/models/img_encoder1.py#L108-L166) 里 `fused_cls` 输出）。不计算 `kan/fusion/fin_cls`，节省算力。
- `head='tab'`：只跑 `kan`，接 `self.tab_cls` 输出。
- `head='fusion'`：保持现状：跑两路→`multiscale_fusion`→`fin_cls`。
- `mode` 仍用于消融（mask 某路特征），主要在 `head='fusion'` 评估时用。

### 2) 增加训练阶段参数（CLI）
在 [train_arg_parser.py](file:///e:/run_our_data/MMCAF-Net-main/args/train_arg_parser.py) 增加：
- `--train_stage`：`img_pretrain | tab_pretrain | fusion_train | full`（默认 `full`，保持老行为）。
- `--resume_optimizer`：是否从 `--ckpt_path` 恢复 optimizer/scheduler（默认 `false`，因为分阶段切换可训练参数集合后，旧 optimizer state 很容易不匹配）。

### 3) train1.py 按阶段执行冻结与参数选择
在 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py) 的“模型构建/加载权重”之后、创建 optimizer 之前加入：
- 一个 stage→head 的映射：
  - `img_pretrain → head='img'`
  - `tab_pretrain → head='tab'`
  - `fusion_train → head='fusion'`
  - `full → head='fusion'`
- 冻结策略（都作用在 `model.module` 上，因为外层是 `DataParallel`）：
  - `img_pretrain`：仅 `image_encoder` 参与训练；冻结 `kan/multiscale_fusion/fin_cls/tab_cls`。
  - `tab_pretrain`：仅 `kan + tab_cls` 参与训练；冻结 `image_encoder/multiscale_fusion/fin_cls`。
  - `fusion_train`：仅 `multiscale_fusion + fin_cls` 参与训练；冻结 `image_encoder/kan/tab_cls`。
  - `full`：不冻结。
- optimizer 参数来源改为：只传 `requires_grad=True` 的参数（而不是无脑 `model.parameters()`）。
- 如果 `--ckpt_path` 存在：
  - 始终 load `model_state`（用于阶段间 warm-start）。
  - 只有当 `--resume_optimizer=true` 时才调用 `ModelSaver.load_optimizer(...)`。
- 训练循环里前向改为：`out = model.forward(img, tab, head=train_head)`。

### 4) 评估器支持 head（保证每个阶段评估用对输出）
修改 [model_evaluator1.py](file:///e:/run_our_data/MMCAF-Net-main/evaluator/model_evaluator1.py) ：
- 给 `evaluate(...)` / `_eval_phase(...)` 增加可选参数 `head='fusion'`。
- 调用点从 `model.forward(img, tab, mode=forward_mode)` 改为 `model.forward(img, tab, mode=forward_mode, head=head)`。
- 在 [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py) 调用 `evaluator.evaluate(...)` 时传入当前阶段 head。

## 你将如何使用（建议的三次训练命令）
说明：按你的要求，本机只做调试；真正训练请在服务器（L20）上运行。这里仅给出命令形态（使用你指定解释器）。
- 阶段1（图像预训练）：`--train_stage img_pretrain`（不恢复 optimizer）
- 阶段2（表格预训练）：`--train_stage tab_pretrain --ckpt_path <阶段1的best.pth.tar>`
- 阶段3（融合+分类器）：`--train_stage fusion_train --ckpt_path <阶段2的best.pth.tar>`

## 验证方式（只做本机快速 smoke，不训练）
实现完成后我会做两类验证：
- 静态：能正常 import，CLI 能解析 `--train_stage`。
- 动态（极小 batch/随机张量）：分别跑 `head=img/tab/fusion` 的 forward，确认形状都是 `[B,1]`，并确认冻结后 `requires_grad`/optimizer param 数量符合预期。

## 涉及文件（将会改动）
- [MMCAF_Net.py](file:///e:/run_our_data/MMCAF-Net-main/models/MMCAF_Net.py)
- [train_arg_parser.py](file:///e:/run_our_data/MMCAF-Net-main/args/train_arg_parser.py)
- [train1.py](file:///e:/run_our_data/MMCAF-Net-main/train1.py)
- [model_evaluator1.py](file:///e:/run_our_data/MMCAF-Net-main/evaluator/model_evaluator1.py)

如果你确认这个方案，我会按上述步骤直接落代码，并在本机做不涉及训练的 smoke 验证，保证三阶段跑起来且不影响现有 full 训练/测试流程。