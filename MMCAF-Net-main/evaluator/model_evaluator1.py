import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util
import warnings
from sklearn.impute import KNNImputer

# 屏蔽无关警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.helpers is deprecated")

from tqdm import tqdm
from .output_aggregator import OutputAggregator
from cams.grad_cam import GradCAM

###
import shap
import cv2
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import gc

class ModelEvaluator1(object):
    def __init__(self, 
                 dataset_name, 
                 data_loaders, 
                 agg_method = None, 
                 epochs_per_eval = 1):


        self.aggregator=OutputAggregator(agg_method, num_bins=10, num_epochs=5)
        
        self.data_loaders=data_loaders
        self.dataset_name=dataset_name
        self.epochs_per_eval=epochs_per_eval #w
        self.cls_loss_fn= util.optim_util.get_loss_fn(is_classification=True, dataset=dataset_name)
        self.max_eval=None 



    def evaluate(self, model, device, epoch=None, num_epochs=None, table=None):

        #w
        # tab = pd.read_csv('e:/rerun2/data/G_first_last_nor.csv')
        import os
        if table is not None:
            tab = table
        else:
            # 自动从 dataset 获取数据目录，避免硬编码路径错误
            data_dir = self.data_loaders[0].dataset.data_dir
            csv_path = os.path.join(data_dir, 'G_first_last_nor.csv')
            tab = pd.read_csv(csv_path)
            # 已在预处理阶段完成 KNN 插值和归一化，直接使用
        
        metrics, curves={}, {}

        # Grad-CAM 初始化 (只在验证集第一个样本做一次即可，节省时间)
        if not hasattr(self, 'grad_cam'):
             self.grad_cam = GradCAM(model, device, is_binary=True, is_3d=True)
        
        # 目标层：根据 MMCAF_Net 结构，选择图像编码器最后一层特征融合/输出前的层
        # models/MMCAF_Net.py: self.image_encoder = Img_new()
        # models/img_encoder1.py: Img_new 内部包含 self.bfpu2 / self.dcfb1 等层
        # 因此常用目标层应为: image_encoder.bfpu2 / image_encoder.dcfb1
        model_to_scan = model.module if hasattr(model, 'module') else model
        available_layer_names = set(name for name, _ in model_to_scan.named_modules())
        target_layer_candidates = [
            'image_encoder.bfpu2',
            'image_encoder.dcfb1',
            'image_encoder.dcfb2',
            'image_encoder.dcfb3',
        ]
        target_layer = None
        for cand in target_layer_candidates:
            if cand in available_layer_names:
                target_layer = cand
                break
        if target_layer is None:
            print("WARNING: Could not resolve Grad-CAM target layer from candidates; skipping Grad-CAM.")
            target_layer = ''

        #w 还不确定self.data_loaders是不是有多个元素
        sum_loss = []

        model.eval()
        for data_loader in self.data_loaders:
            # Determine if we should perform heavy analysis (SHAP, Grad-CAM, Ablation)
            do_analysis = False
            if epoch is not None:
                # 修改为每 1 Epoch 执行一次深度分析，方便用户立即看到结果
                # 原本是 2
                if epoch % 1 == 0 or (num_epochs is not None and epoch >= num_epochs):
                    do_analysis = True
            
            phase_metrics, phase_curves, sum_every_loss = self._eval_phase(
                model, data_loader, data_loader.phase, device, tab, epoch, do_analysis, target_layer
            )
            metrics.update(phase_metrics)
            curves.update(phase_curves)
            #w
            sum_loss.append(sum_every_loss)
            
        model.train()
        #w
        eval_loss = sum(sum_loss) / len(sum_loss)
        # raise ValueError("eval_loss是{}".format(eval_loss))
        print('eval_loss:', eval_loss)
        ###
        return metrics,curves, eval_loss

    ###
    def _eval_phase(self, model, data_loader, phase, device, table, epoch, do_analysis=False, target_layer='image_encoder.bfpu2'):
        #w
        out = None
        phase_curves = {} # 初始化用于存储分析结果（如 Grad-CAM）的字典


        # 单模态屏蔽评估 (Ablation Study) - 只在 Val/Test 且满足 do_analysis 条件时进行
        ablation_metrics = {}
        if phase in ['val', 'test'] and do_analysis:
            print(f"Running Ablation Study for {phase} (Epoch {epoch})...")
            
            for mode in ['img_only', 'tab_only', 'baseline']:
                mode_metrics, mode_curves = self._run_ablation(model, data_loader, device, table, mode)
                
                # 为指标和曲线添加后缀，方便在 TensorBoard 中对比
                suffix = {
                    'img_only': 'ImgOnly',
                    'tab_only': 'TabOnly',
                    'baseline': 'Multimodal'
                }[mode]
                
                # 记录核心指标 (仅保留 Loss 和 AUROC，精简 Scalars 显示)
                target_metrics = ['loss', 'AUROC']
                for k, v in mode_metrics.items():
                    metric_name = k.split('_')[-1]
                    if metric_name in target_metrics:
                        ablation_metrics[f'{phase}_Ablation_{suffix}/{metric_name}'] = v
                
                # 记录混淆矩阵曲线 (仅保留混淆矩阵)
                for k, v in mode_curves.items():
                    if 'Confusion Matrix' in k:
                        # 命名格式必须符合 BaseLogger 的后缀识别逻辑 (以 _Confusion Matrix 结尾)
                        phase_curves[f'{phase}_Ablation_{suffix}_Confusion Matrix'] = v
                    # 其他曲线 (PRC, ROC) 在此被忽略，不写入 phase_curves
        
        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
        """
        batch_size=data_loader.batch_size

        # Keep track of task-specific records needed for computing overall metrics
    
        records={'keys': [], 'probs': []}
        


     
        num_examples=len(data_loader.dataset)
      

        # Sample from the data loader and record model outputs
        num_evaluated=0

        with tqdm(total=num_examples, unit=' ' + phase) as progress_bar:
            #w
            sum_every_loss = 0

            for img, targets_dict in data_loader:
                if num_evaluated >=num_examples:
                    break

                ###
                ids = [item for item in targets_dict['study_num']]

                # 动态识别特征列
                metadata_cols = ['NewPatientID', 'OriginalIndex', 'label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
                feature_cols = [c for c in table.columns if c not in metadata_cols and not c.startswith('Unnamed')]
                
                tab=[]
                for i in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[i]]
                    if patient_row.empty:
                        data = np.zeros(len(feature_cols), dtype=np.float32)
                    else:
                        data = patient_row[feature_cols].iloc[0].values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab)
                if tab.ndim == 3: # 处理可能的维度问题
                    tab = tab.squeeze(1)

                with torch.no_grad():
                    #w process data
                    img = img.to(device)
                    tab = tab.to(device)
                    # 修正：BCEWithLogitsLoss 要求 label 为 Float 类型
                    label = targets_dict['is_abnormal'].to(device).float()

                    # 使用自动混合精度 (AMP) 评估，保持与训练一致
                    with torch.cuda.amp.autocast():
                        out = model.forward(img, tab)
                        label = label.unsqueeze(1)
                        cls_loss = self.cls_loss_fn(out, label).mean()
                        loss = cls_loss
                    
                    #w
                    sum_every_loss += loss.item()
                    cls_logits = out if out is not None else torch.randn([img.size(0), 1]).to(device)


                    



                #w
                self._record_batch(cls_logits,targets_dict['series_idx'],loss,**records)

                # 可解释性分析 (仅在验证集每个epoch的第一个batch做一次)
                # 优化：只在满足 do_analysis 条件的 Epoch 进行耗时的可视化分析 (Grad-CAM, AttnMap, SHAP)
                if phase == 'val' and num_evaluated == 0 and do_analysis:
                    print(f"DEBUG: Starting heavy analysis (Grad-CAM, SHAP, etc.) for {phase} at Epoch {epoch}...")
                    # 1. Grad-CAM 可视化
                    try:
                        # 目标层：指向图像编码器最后的特征提取层
                        target_layer = target_layer 
                        print(f"DEBUG: Running Grad-CAM on layer: {target_layer}")
                        with torch.enable_grad():
                             if hasattr(self, 'grad_cam') and target_layer:
                                 self.grad_cam.model = model 
                                 self.grad_cam._register_hooks(target_layer)
                                 idx = 0 
                                 # 显式 clone 输入，防止干扰
                                 cam_img = img[idx:idx+1].detach().clone().requires_grad_(True)
                                 cam_tab = tab[idx:idx+1].detach().clone()
                                 
                                 probs, _ = self.grad_cam.forward(cam_img, cam_tab)
                                 self.grad_cam.backward(idx=0)
                                 gcam = self.grad_cam.get_cam(target_layer)
                                 phase_curves[f'{phase}_GradCAM'] = gcam
                                 
                                 # 立即释放钩子并清理梯度
                                 self.grad_cam._release_hooks()
                                 model.zero_grad()
                                 print("DEBUG: Grad-CAM completed and hooks released.")
                    except Exception as e:
                        print(f"ERROR: Grad-CAM failed: {e}")
                        if hasattr(self, 'grad_cam'): self.grad_cam._release_hooks()
                    finally:
                        # 确保无论成功失败都清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 2. Attention Matrix 可视化
                    try:
                        if hasattr(model, 'multiscale_fusion'):
                            ca_module = model.multiscale_fusion.cross_attention1
                            if hasattr(ca_module, 'last_attn_map'):
                                print("DEBUG: Logging Attention Map...")
                                attn_map = ca_module.last_attn_map[0, 0] 
                                phase_curves[f'{phase}_AttnMap'] = attn_map.detach().cpu().numpy()
                    except Exception as e:
                        print(f"ERROR: AttnMap log failed: {e}")

                    # 3. SHAP Summary Plot 可视化（使用 KernelExplainer）
                    try:
                        print("DEBUG: Running SHAP (KernelExplainer) for tabular features...")

                        # 动态获取特征名称
                        metadata_cols = ['NewPatientID', 'OriginalIndex', 'label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
                        current_feature_names = [c for c in table.columns if c not in metadata_cols and not c.startswith('Unnamed')]
                        
                        # 控制 KernelExplainer 的计算量
                        bg_n = min(8, tab.shape[0])
                        ex_n = min(2, tab.shape[0])

                        x_background = tab[:bg_n].detach().cpu().numpy()
                        x_explain = tab[:ex_n].detach().cpu().numpy()

                        img_fixed = img[:1].detach().clone()

                        def _predict_tab(x_np):
                            x_np = np.asarray(x_np, dtype=np.float32)
                            if x_np.ndim == 1:
                                x_np = x_np[None, :]
                            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)

                            img_fixed_dev = img_fixed.to(device)

                            # 分批推理
                            chunk = 2
                            all_probs = []
                            with torch.inference_mode():
                                for start in range(0, x_t.shape[0], chunk):
                                    end = min(start + chunk, x_t.shape[0])
                                    x_chunk = x_t[start:end]
                                    img_rep = img_fixed_dev.repeat(x_chunk.shape[0], 1, 1, 1, 1)
                                    if torch.cuda.is_available():
                                        with torch.cuda.amp.autocast(dtype=torch.float16):
                                            logits = model.forward(img_rep, x_chunk)
                                    else:
                                        logits = model.forward(img_rep, x_chunk)
                                    probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
                                    all_probs.append(probs)
                            return np.concatenate(all_probs, axis=0)

                        explainer = shap.KernelExplainer(_predict_tab, x_background)
                        shap_values = explainer.shap_values(x_explain, nsamples=50)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]

                        # 显式创建 Figure 并设置大小
                        fig = plt.figure(figsize=(12, 8))
                        shap.summary_plot(
                            shap_values,
                            x_explain,
                            feature_names=current_feature_names,
                            show=False,
                            plot_type="bar"
                        )
                        # 强制重绘
                        plt.tight_layout()
                        
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        shap_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                        shap_img = cv2.imdecode(shap_img, cv2.IMREAD_COLOR)
                        if shap_img is not None:
                            # shap_img 是 (H, W, C)，需要保持 HWC 格式，不要转置
                            # BaseLogger 会在 add_image 时自动处理 HWC -> CHW
                            # 这里只需确保它是 RGB
                            shap_img = cv2.cvtColor(shap_img, cv2.COLOR_BGR2RGB)
                            phase_curves[f'{phase}_SHAP_Summary'] = shap_img
                            print(f"DEBUG: SHAP summary plot completed with {len(current_feature_names)} features.")
                        else:
                            print("ERROR: SHAP image decode failed!")

                        plt.close(fig) 
                        buf.close()
                        del explainer
                    except Exception as e:
                        print(f"ERROR: SHAP plot failed: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                progress_bar.update(min(batch_size, num_examples - num_evaluated))
                num_evaluated +=batch_size



        #Map to summary dictionaries
        metrics, curves = self._get_summary_dicts(data_loader, phase, device, **records)
        metrics.update(ablation_metrics) # 合并消融实验结果
        curves.update(phase_curves)      # 合并分析结果图

        ###
        return metrics, curves, sum_every_loss

    def _run_ablation(self, model, data_loader, device, table, mode):
        """快速运行单模态消融评估，返回完整指标和曲线"""
        records = {'keys': [], 'probs': []}
        
        with torch.no_grad():
            for i, (img, targets_dict) in enumerate(data_loader):
                # 临床特征处理逻辑
                ids = [item for item in targets_dict['study_num']]
                metadata_cols = ['NewPatientID', 'OriginalIndex', 'label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
                feature_cols = [c for c in table.columns if c not in metadata_cols and not c.startswith('Unnamed')]

                tab=[]
                for k in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[k]]
                    if patient_row.empty: 
                        data = np.zeros(len(feature_cols), dtype=np.float32)
                    else: 
                        data = patient_row[feature_cols].iloc[0].values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab)
                if tab.ndim == 3:
                    tab = tab.squeeze(1)

                img = img.to(device)
                tab = tab.to(device)
                series_indices = targets_dict['series_idx']

                # 实施屏蔽
                if mode == 'img_only':
                    tab = torch.zeros_like(tab)
                elif mode == 'tab_only':
                    img = torch.zeros_like(img)

                # 使用自动混合精度 (AMP) 评估
                with torch.cuda.amp.autocast():
                    out = model.forward(img, tab)
                
                self._record_batch(out, series_indices, None, **records)
        
        # 计算该模式下的所有指标和曲线
        metrics, curves = self._get_summary_dicts(data_loader, mode, device, **records)
        model.zero_grad()
        return metrics, curves

    @staticmethod
    def _record_batch(logits, targets, loss, probs=None, keys=None, loss_meter=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            targets: Batch of ground-truth targets corresponding to the logits.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if probs is not None:
            assert keys is not None, 'Must keep probs and keys lists in parallel'
            with torch.no_grad():
                batch_probs=F.sigmoid(logits)
            probs.append(batch_probs.detach().cpu())

            #Note: `targets` is assumed to hold the keys for these examples
            keys.append(targets.detach().cpu())
        

    def _get_summary_dicts(self, data_loader, phase, device, probs=None, keys=None, loss_meter=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model. E.g. ROC.
        """
        metrics, curves={}, {}

        if probs is not None:
            # If records kept track of individual probs and keys, implied that we need to aggregate them
            assert keys is not None, 'Must keep probs and keys lists in parallel.'
            assert self.aggregator is not None, 'Must specify an aggregator to aggregate probs and keys.'

            # Convert to flat numpy array
            probs=np.concatenate(probs).ravel().tolist()
            keys=np.concatenate(keys).ravel().tolist()

            # Aggregate predictions across each series
            idx2prob=self.aggregator.aggregate(keys, probs, data_loader, phase, device)
            probs, labels=[], []
            for idx, prob in idx2prob.items():
                probs.append(prob)
                labels.append(data_loader.get_series_label(idx))
            probs, labels=np.array(probs), np.array(labels)

            # Update summary dicts
            try:
                # 检查是否存在 NaN
                if np.isnan(probs).any():
                    raise ValueError("Model output contains NaN")

                metrics.update({
                    phase + '_' + 'loss': sk_metrics.log_loss(labels, probs, labels=[0, 1])
                })

                # Binarize predictions for Acc, Sens, Spec
                preds = (probs >= 0.5).astype(int)
                
                # Calculate metrics
                accuracy = sk_metrics.accuracy_score(labels, preds)
                conf_matrix = sk_metrics.confusion_matrix(labels, preds, labels=[0, 1])
                tn, fp, fn, tp = conf_matrix.ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                f1 = sk_metrics.f1_score(labels, preds)

                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                    phase + '_' + 'Accuracy': accuracy,
                    phase + '_' + 'Sensitivity': sensitivity,
                    phase + '_' + 'Specificity': specificity,
                    phase + '_' + 'PPV': precision,
                    phase + '_' + 'NPV': npv,
                    phase + '_' + 'F1': f1
                })
                curves.update({
                    phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, probs),
                    phase + '_' + 'ROC': sk_metrics.roc_curve(labels, probs),
                    phase + '_' + 'Confusion Matrix': sk_metrics.confusion_matrix(labels, preds)
                })
            except ValueError as e:
                print(f"CRITICAL: {phase} evaluation failed due to NaN or invalid values: {e}")
                # 设置极端指标，让用户在 TensorBoard 中能看到明显的“崩溃”信号
                metrics.update({
                    phase + '_' + 'loss': 99.0,
                    phase + '_' + 'AUROC': 0.0,
                    phase + '_' + 'Accuracy': 0.0,
                    phase + '_' + 'F1': 0.0
                })



        return metrics, curves
