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

plt.switch_backend('agg')

class ModelEvaluator1(object):
    def __init__(self, 
                 dataset_name, 
                 data_loaders, 
                 agg_method = None, 
                 epochs_per_eval = 1,
                 shap_eval_freq = 5,
                 ablation_eval_freq = 5):


        self.aggregator=OutputAggregator(agg_method, num_bins=10, num_epochs=5)
        
        self.data_loaders=data_loaders
        self.dataset_name=dataset_name
        self.epochs_per_eval=epochs_per_eval #w
        self.shap_eval_freq = shap_eval_freq
        self.ablation_eval_freq = ablation_eval_freq
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
            do_shap = False
            do_ablation = False
            
            if epoch is not None:
                # 判断是否执行 SHAP/Grad-CAM
                if self.shap_eval_freq > 0 and (epoch % self.shap_eval_freq == 0 or (num_epochs is not None and epoch >= num_epochs)):
                    do_shap = True
                
                # 判断是否执行 Ablation
                if self.ablation_eval_freq > 0 and (epoch % self.ablation_eval_freq == 0 or (num_epochs is not None and epoch >= num_epochs)):
                    do_ablation = True
            
            phase_metrics, phase_curves, sum_every_loss = self._eval_phase(
                model, data_loader, data_loader.phase, device, tab, epoch, num_epochs, do_shap, do_ablation, target_layer
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
    def _eval_phase(self, model, data_loader, phase, device, table, epoch, num_epochs=None, do_shap=False, do_ablation=False, target_layer='image_encoder.bfpu2'):
        #w
        out = None
        phase_curves = {} # 初始化用于存储分析结果（如 Grad-CAM）的字典


        # 单模态屏蔽评估 (Ablation Study) - 只在 Val/Test 且满足 do_analysis 条件时进行
        ablation_metrics = {}
        if phase in ['val', 'test'] and do_ablation:
            print(f"Running Ablation Study for {phase} (Epoch {epoch})...")
            
            # 用于存储所有模式的所有指标，生成详细表格
            ablation_results = {}
            
            for mode in ['img_only', 'tab_only', 'baseline']:
                mode_metrics, mode_curves = self._run_ablation(model, data_loader, device, table, mode)
                
                # 为指标和曲线添加后缀，方便在 TensorBoard 中对比
                suffix = {
                    'img_only': 'ImgOnly',
                    'tab_only': 'TabOnly',
                    'baseline': 'Multimodal'
                }[mode]
                
                # 初始化结果字典
                ablation_results[suffix] = {}
                
                # 记录核心指标 (Scalar显示部分保持精简)
                target_scalar_metrics = ['loss', 'AUROC']
                for k, v in mode_metrics.items():
                    metric_name = k.split('_')[-1]
                    
                    # 收集所有指标用于表格
                    # 修正：Loss 首字母大写不一致导致表格显示 -1
                    display_name = 'Loss' if metric_name == 'loss' else metric_name
                    ablation_results[suffix][display_name] = float(v)

                    # 过滤不需要的特定组合 (Scalar)
                    if metric_name == 'loss' and mode in ['tab_only', 'baseline']:
                         continue
                    if metric_name == 'AUROC' and mode == 'img_only':
                         continue
                         
                    if metric_name in target_scalar_metrics:
                        ablation_metrics[f'{phase}_Ablation_{suffix}/{metric_name}'] = v
                
                # 记录混淆矩阵曲线
                for k, v in mode_curves.items():
                    if 'Confusion Matrix' in k:
                        phase_curves[f'{phase}_Ablation_{suffix}_Confusion Matrix'] = v

            # 生成详细对比表格
            if len(ablation_results) > 0:
                # 定义要显示的列
                columns = ['Loss', 'AUROC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity']
                
                # 构建 Markdown 表头
                header = "| Mode | " + " | ".join(columns) + " |\n"
                header += "|---| " + " | ".join(["---:"] * len(columns)) + " |\n"
                
                rows = []
                for suffix in ["ImgOnly", "TabOnly", "Multimodal"]:
                    if suffix in ablation_results:
                        res = ablation_results[suffix]
                        row_str = f"| **{suffix}** |"
                        for col in columns:
                            val = res.get(col, -1)
                            # 根据指标类型格式化
                            if col == 'Loss':
                                row_str += f" {val:.4f} |"
                            else:
                                row_str += f" {val:.4f} |"
                        rows.append(row_str)
                
                phase_curves[f"{phase}_Ablation_Table"] = header + "\n".join(rows)
        
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
                desired_feature_cols = ['实性成分大小', '毛刺征', '支气管异常征', '胸膜凹陷征', 'CEA']
                feature_cols = [c for c in desired_feature_cols if c in table.columns]
                if len(feature_cols) != len(desired_feature_cols):
                    for c in desired_feature_cols:
                        if c not in table.columns:
                            table[c] = 0.0
                    feature_cols = desired_feature_cols
                
                tab=[]
                for i in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[i]]
                    if patient_row.empty:
                        data = np.zeros(len(feature_cols), dtype=np.float32)
                    else:
                        data = patient_row[feature_cols].iloc[0].fillna(0.0).values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab)
                if tab.ndim == 3: # 处理可能的维度问题
                    tab = tab.squeeze(1)

                with torch.no_grad():
                    #w process data
                    img = img.to(device)
                    tab = tab.to(device)

                    # 安全检查：防止输入 NaN
                    if torch.isnan(img).any():
                        print(f"WARNING: Found NaN in input image (batch at {num_evaluated}). Replacing with 0.")
                        img = torch.nan_to_num(img, nan=0.0)
                    if torch.isnan(tab).any():
                        print(f"WARNING: Found NaN in input tabular data (batch at {num_evaluated}). Replacing with 0.")
                        tab = torch.nan_to_num(tab, nan=0.0)

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

                # 可视化输入样本 (检查 BBox 裁剪效果) - 每个 Validation Epoch 做一次
                if phase == 'val' and num_evaluated == 0:
                     self._visualize_inputs(img, phase, phase_curves)

                # 可解释性分析 (仅在验证集每个epoch的第一个batch做一次)
                # 优化：只在满足 do_analysis 条件的 Epoch 进行耗时的可视化分析 (Grad-CAM, AttnMap, SHAP)
                if phase == 'val' and num_evaluated == 0 and do_shap:
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
                                 gcam_t = torch.from_numpy(gcam).unsqueeze(0).unsqueeze(0).float()
                                 gcam_t = F.interpolate(
                                     gcam_t,
                                     size=cam_img.shape[-3:],
                                     mode="trilinear",
                                     align_corners=False,
                                 )
                                 gcam_up = gcam_t.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()
                                 gcam_depth_scores = gcam_up.reshape(gcam_up.shape[0], -1).mean(axis=1)
                                 gcam_slice_idx = int(np.argmax(gcam_depth_scores))
                                 gcam_slice = gcam_up[gcam_slice_idx]
                                 phase_curves[f'{phase}_GradCAM'] = gcam_slice
                                 
                                 # --- 优化 CT 图像归一化显示 ---
                                 ct_slice = cam_img[0, 0, gcam_slice_idx].detach().cpu().numpy().astype(np.float32)
                                 # 移除可能的 NaN/Inf
                                 ct_slice = np.nan_to_num(ct_slice, nan=0.0, posinf=0.0, neginf=0.0)
                                 
                                 # 使用 CLAHE 增强 CT 底图对比度
                                 ct_min, ct_max = ct_slice.min(), ct_slice.max()
                                 if ct_max - ct_min > 1e-6:
                                     ct_norm = (ct_slice - ct_min) / (ct_max - ct_min)
                                 else:
                                     ct_norm = ct_slice
                                 ct_u8 = (ct_norm * 255.0).astype(np.uint8)
                                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                 ct_enhanced = clahe.apply(ct_u8)
                                 ct_rgb = cv2.cvtColor(ct_enhanced, cv2.COLOR_GRAY2RGB)
 
                                 # --- 优化 Overlay 逻辑 (Screen Blending + Threshold) ---
                                 heat_u8 = (np.clip(gcam_slice, 0.0, 1.0) * 255.0).astype(np.uint8)
                                 heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                                 heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
                                 
                                 threshold = 0.2
                                 mask_val = np.maximum(0, gcam_slice - threshold) / (1.0 - threshold)
                                 mask_val = cv2.GaussianBlur(mask_val, (3, 3), 0)
                                 alpha = mask_val[..., None]
                                 
                                 ct_float = ct_rgb.astype(np.float32) / 255.0
                                 heat_float = heat_rgb.astype(np.float32) / 255.0
                                 blend_screen = 1.0 - (1.0 - ct_float) * (1.0 - heat_float)
                                 alpha_limited = np.clip(alpha, 0, 0.7)
                                 out_float = ct_float * (1.0 - alpha_limited) + blend_screen * alpha_limited
                                 overlay = (np.clip(out_float, 0, 1) * 255.0).astype(np.uint8)
                                 
                                 phase_curves[f'{phase}_GradCAM_Overlay'] = overlay
                                 
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

                        desired_feature_cols = ['实性成分大小', '毛刺征', '支气管异常征', '胸膜凹陷征', 'CEA']
                        current_feature_names = [c for c in desired_feature_cols if c in table.columns]
                        if len(current_feature_names) != len(desired_feature_cols):
                            for c in desired_feature_cols:
                                if c not in table.columns:
                                    table[c] = 0.0
                            current_feature_names = desired_feature_cols
                        
                        unique_table = table.drop_duplicates(subset=['NewPatientID'], keep='first')
                        x_pool = unique_table[current_feature_names].to_numpy(dtype=np.float32, copy=True)
                        x_pool = np.nan_to_num(x_pool, nan=0.0, posinf=0.0, neginf=0.0)

                        pool_n = int(x_pool.shape[0])
                        if pool_n < 2 or len(current_feature_names) == 0:
                            raise ValueError(f"Insufficient tabular samples/features for SHAP: pool_n={pool_n}, features={len(current_feature_names)}")

                        is_final_epoch = (num_epochs is not None and epoch is not None and epoch >= num_epochs)

                        bg_n = min(10, pool_n)
                        ex_n = min(5, pool_n)

                        if is_final_epoch:
                            print("DEBUG: Final epoch detected. Running high-quality SHAP on full data...")
                            bg_n = min(100, pool_n)
                            ex_n = pool_n

                        bg_idx = np.random.choice(pool_n, size=bg_n, replace=False)
                        ex_idx = np.random.choice(pool_n, size=ex_n, replace=False)
                        x_background = x_pool[bg_idx]
                        x_explain = x_pool[ex_idx]

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
                        nsamples = 200 if is_final_epoch else 50
                        shap_values = explainer.shap_values(x_explain, nsamples=nsamples)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]

                        shap_values = np.asarray(shap_values, dtype=np.float32)
                        if shap_values.ndim == 1:
                            shap_values = shap_values.reshape(1, -1)
                        if shap_values.shape[-1] != len(current_feature_names):
                            raise ValueError(f"SHAP shape mismatch: shap_values={shap_values.shape}, features={len(current_feature_names)}")

                        shap.summary_plot(
                            shap_values,
                            x_explain,
                            feature_names=current_feature_names,
                            show=False,
                            plot_type="dot",
                            max_display=len(current_feature_names)
                        )
                        fig = plt.gcf()
                        fig.set_size_inches(12, 8)
                        fig.tight_layout()
                        fig.canvas.draw()
                        w, h = fig.canvas.get_width_height()
                        shap_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                        phase_curves[f'{phase}_SHAP_Summary'] = shap_img
                        print(f"DEBUG: SHAP summary plot completed with {len(current_feature_names)} features.")

                        plt.close(fig)
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
        
        # 映射 mode 字符串到整数 flag
        # mode: 0=multimodal, 1=img_only (mask tab), 2=tab_only (mask img)
        mode_flag = 0
        if mode == 'img_only':
            mode_flag = 1
        elif mode == 'tab_only':
            mode_flag = 2
            
        with torch.no_grad():
            for i, (img, targets_dict) in enumerate(data_loader):
                # 临床特征处理逻辑
                ids = [item for item in targets_dict['study_num']]
                desired_feature_cols = ['实性成分大小', '毛刺征', '支气管异常征', '胸膜凹陷征', 'CEA']
                feature_cols = [c for c in desired_feature_cols if c in table.columns]
                if len(feature_cols) != len(desired_feature_cols):
                    for c in desired_feature_cols:
                        if c not in table.columns:
                            table[c] = 0.0
                    feature_cols = desired_feature_cols

                tab=[]
                for k in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[k]]
                    if patient_row.empty: 
                        data = np.zeros(len(feature_cols), dtype=np.float32)
                    else: 
                        data = patient_row[feature_cols].iloc[0].fillna(0.0).values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab)
                if tab.ndim == 3:
                    tab = tab.squeeze(1)

                img = img.to(device)
                tab = tab.to(device)
                
                # 安全检查：防止输入 NaN 导致模型输出 NaN
                if torch.isnan(img).any():
                    print(f"WARNING: Found NaN in input image (batch {i}). Replacing with 0.")
                    img = torch.nan_to_num(img, nan=0.0)
                
                if torch.isnan(tab).any():
                    print(f"WARNING: Found NaN in input tabular data (batch {i}). Replacing with 0.")
                    tab = torch.nan_to_num(tab, nan=0.0)

                series_indices = targets_dict['series_idx']

                # 注意：不再修改输入 img/tab，而是在 forward 中通过 mode_flag 屏蔽特征
                # 这能避免全0输入导致的 BatchNorm 偏移或 Encoder Bias 问题

                # 使用自动混合精度 (AMP) 评估
                with torch.cuda.amp.autocast():
                    out = model.forward(img, tab, mode=mode_flag)
                
                if torch.isnan(out).any():
                    print(f"CRITICAL WARNING: Model output contains NaN at batch {i} (Mode={mode})!")
                
                self._record_batch(out, series_indices, None, **records)
        
        # 计算该模式下的所有指标和曲线
        metrics, curves = self._get_summary_dicts(data_loader, mode, device, **records)
        model.zero_grad()
        return metrics, curves

    def _visualize_inputs(self, img, phase, curves):
        """可视化输入 Batch 的中间切片，用于验证数据预处理（如 BBox 裁剪）效果"""
        try:
            # img shape: (B, C, D, H, W)
            # 取前 4 个样本
            num_samples = min(img.shape[0], 4)
            samples = img[:num_samples].detach().cpu().numpy()
            
            grid_images = []
            for i in range(num_samples):
                # 取中间切片: samples[i, 0, D//2, :, :]
                # 假设 Channel 0 是 CT 图像
                d_idx = samples.shape[2] // 2
                mid_slice = samples[i, 0, d_idx, :, :]
                
                # 归一化到 0-255
                # 注意：输入图像可能已经被归一化过（例如 -1 到 1 或 0 到 1），也可能包含负值
                # 这里使用 Min-Max 归一化到 0-255 用于显示
                min_val, max_val = mid_slice.min(), mid_slice.max()
                if max_val - min_val > 1e-6:
                    norm_slice = (mid_slice - min_val) / (max_val - min_val)
                else:
                    norm_slice = np.zeros_like(mid_slice) # 避免全白/全黑
                
                img_u8 = (norm_slice * 255).astype(np.uint8)
                # 转为 RGB
                img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
                
                # 添加文字说明 (Sample Index)
                # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
                cv2.putText(img_rgb, f"S{i}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                grid_images.append(img_rgb)
            
            # 拼接成一行
            if grid_images:
                # 简单水平拼接
                grid_combined = np.hstack(grid_images)
                curves[f'{phase}_Input_Samples'] = grid_combined
                print(f"DEBUG: Generated input samples visualization for {phase}.")
                
        except Exception as e:
            print(f"ERROR: Failed to visualize inputs: {e}")

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
                
                # 避免除零错误
                denominator_sens = tp + fn
                denominator_spec = tn + fp

                sensitivity = tp / denominator_sens if denominator_sens > 0 else 0.0
                specificity = tn / denominator_spec if denominator_spec > 0 else 0.0
                f1 = sk_metrics.f1_score(labels, preds)

                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                    phase + '_' + 'Accuracy': accuracy,
                    phase + '_' + 'Sensitivity': sensitivity,
                    phase + '_' + 'Specificity': specificity,
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
