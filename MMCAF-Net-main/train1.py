
import models
import torch
import torch.nn as nn
import util
import warnings
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import numpy as np

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)# 1. 锁定 Python 内置随机数（影响 shuffle 等）
    np.random.seed(seed)# 2. 锁定 NumPy 随机数（影响数据预处理）
    torch.manual_seed(seed)# 3. 锁定 PyTorch CPU 随机数（影响模型初始化）
    torch.cuda.manual_seed(seed)# 4. 锁定 PyTorch GPU 随机数（影响模型初始化）
    torch.cuda.manual_seed_all(seed)# 5. 锁定所有 GPU 随机数（影响模型初始化）
    torch.backends.cudnn.deterministic = True# 6. 启用确定性卷积算法（影响性能）
    torch.backends.cudnn.benchmark = False# 7. 禁用卷积算法自动选择（影响性能）
    os.environ['PYTHONHASHSEED'] = str(seed)# 8. 锁定 Python 哈希种子（影响 shuffle 等）    

set_seed(42)

# 屏蔽无关警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.helpers is deprecated")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore" # 强制环境变量屏蔽

from torch.autograd import Variable

from args import TrainArgParser
from evaluator import ModelEvaluator1
from logger import TrainLogger
from saver import ModelSaver
from pkl_read import CTPE

###
import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

#w
import data_loader

from PIL import Image
def save_2d_slice(data, count):
    data = data.detach().cpu().numpy()
    output_path = str(count) + '.png'
    slice_index = data.shape[2] // 2
    slice_2d = data[0, 0, slice_index, :, :]
    slice_2d = slice_2d.astype(np.float32)
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
    slice_2d = slice_2d.astype(np.uint8)
    #w
    img = Image.fromarray(slice_2d, mode = 'L')
    img.save(output_path)


def _get_model_inner(model):
    return model.module if hasattr(model, 'module') else model


def _configure_branch_training(model, args):
    train_mode = getattr(args, 'train_mode', 'multimodal')
    scope = getattr(args, 'single_modal_train_scope', 'encoder_head')
    mode_flag = {'multimodal': 0, 'img_only': 1, 'tab_only': 2}.get(train_mode, 0)
    args.forward_mode_flag = mode_flag

    m = _get_model_inner(model)

    if train_mode == 'multimodal':
        for p in m.parameters():
            p.requires_grad = True
        m.train()
        return

    # Unified Strategy: Freeze All First
    for p in m.parameters():
        p.requires_grad = False
    
    # Set global eval (freezes BN stats etc), then unfreeze specific modules
    m.eval()

    def activate(module):
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
            module.train()

    def freeze(module):
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

    # Retrieve modules
    image_encoder = getattr(m, 'image_encoder', None)
    kan = getattr(m, 'kan', None)
    multiscale_fusion = getattr(m, 'multiscale_fusion', None)
    fin_cls = getattr(m, 'fin_cls', None)

    if train_mode == 'img_only':
        activate(image_encoder)
        activate(multiscale_fusion)
        if scope == 'encoder_head':
            activate(fin_cls)
        freeze(kan)

    elif train_mode == 'tab_only':
        activate(kan)
        activate(multiscale_fusion)
        if scope == 'encoder_head':
            activate(fin_cls)
        freeze(image_encoder)



def _get_trainable_parameters(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError('No trainable parameters. Check train_mode/scope configuration.')
    return params


def _apply_override_lr(optimizer, lr_scheduler, override_lr):
    if override_lr is None:
        return
    try:
        override_lr = float(override_lr)
    except Exception:
        return
    if override_lr <= 0:
        return
    for pg in optimizer.param_groups:
        pg['lr'] = override_lr
    if lr_scheduler is not None and hasattr(lr_scheduler, 'base_lrs'):
        try:
            lr_scheduler.base_lrs = [override_lr for _ in lr_scheduler.base_lrs]
        except Exception:
            pass


def _get_scheduler_step(logger, args):
    if not getattr(args, 'reset_scheduler', False):
        return logger.global_step
    offset = getattr(args, 'scheduler_step_offset', 0)
    start = int(getattr(args, 'scheduler_start_step', 0) or 0)
    rel = logger.global_step - int(offset)
    if rel < 0:
        rel = 0
    return start + rel



###
def train(args,table):
    desired_feature_cols = ['实性成分大小', '毛刺征', '支气管异常征', '胸膜凹陷征', 'CEA']
    feature_cols = [c for c in desired_feature_cols if c in table.columns]
    missing_cols = [c for c in desired_feature_cols if c not in table.columns]
    if len(missing_cols) > 0:
        print(f"WARNING: Missing tabular features in CSV: {missing_cols}. Filling with 0.")
        for c in missing_cols:
            table[c] = 0.0
        feature_cols = desired_feature_cols
    print(f"Using {len(feature_cols)} tabular features: {feature_cols}")
    
    # --- Forcing Z-Score Normalization ---
    # Even if data is already MinMax scaled [0, 1], Z-Score is better for KAN/Neural Networks
    if len(feature_cols) > 0:
        print("Applying StandardScaler (Z-Score) to clinical features...")
        scaler = StandardScaler()
        table[feature_cols] = scaler.fit_transform(table[feature_cols])
    # -------------------------------------

    # Add to args so model can use it
    args.num_tab_features = len(feature_cols)
    
    count = 0
    much = 50

    if args.ckpt_path and not args.use_pretrained and os.path.exists(args.ckpt_path):
        print(f"Resuming from checkpoint: {args.ckpt_path}")
        # model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        
        # 手动初始化模型以确保参数（如 num_tab_features）与当前数据一致
        # 避免因 Checkpoint 中保存的 args 不准确导致形状不匹配
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
        
        # 加载权重
        try:
            ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        except (RuntimeError, Exception) as e:
            if "failed finding central directory" in str(e) or "zip" in str(e).lower() or "magic number" in str(e).lower():
                print(f"\n{'!'*60}")
                print(f"[Error] Checkpoint file is corrupted: {args.ckpt_path}")
                print(f"Reason: {e}")
                print(f"Action: Please delete this file manually and restart training.")
                print(f"        The script will then automatically resume from the previous epoch.")
                print(f"{'!'*60}\n")
                sys.exit(1)
            else:
                raise e

        model.load_state_dict(ckpt_dict['model_state'])
        
        ckpt_info = ckpt_dict['ckpt_info']
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        ###
        model = model_fn(**vars(args))
        if args.use_pretrained:
            model.load_pretrained(args.ckpt_path, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    _configure_branch_training(model, args)

    if args.use_pretrained or args.fine_tune:
        inner = _get_model_inner(model)
        if not hasattr(inner, 'fine_tuning_parameters'):
            raise AttributeError(f"Model {type(inner).__name__} does not implement fine_tuning_parameters(). Disable --fine_tune/--use_pretrained or switch model.")
        parameters = inner.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
        optimizer = util.get_optimizer(parameters, args)
    else:
        optimizer = util.get_optimizer(_get_trainable_parameters(model), args)
    lr_scheduler = util.get_scheduler(optimizer, args)

    resume_optimizer = getattr(args, 'resume_optimizer', None)
    if resume_optimizer is None:
        resume_optimizer = bool(getattr(args, 'train_mode', 'multimodal') == 'multimodal')

    if args.ckpt_path and not args.use_pretrained and not args.fine_tune and resume_optimizer:
        if getattr(args, 'reset_scheduler', False):
            ModelSaver.load_optimizer(args.ckpt_path, optimizer, None)
        else:
            ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
        # ModelSaver.load_optimizer(args.ckpt_path, optimizer, D_lr_scheduler)
        _apply_override_lr(optimizer, lr_scheduler, getattr(args, 'override_lr', -1.0))

    # Get logger, evaluator, saver
    # 恢复标准 BCE Loss，严格遵循论文设定
    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    
    # 自动混合精度 AMP 设置
    # args.use_amp 是从命令行传入的参数，默认为 True
    use_amp = args.use_amp and torch.cuda.is_available()
    
    if args.use_amp and not torch.cuda.is_available():
        print("Warning: AMP requested but CUDA is not available. AMP will be disabled.")

    bf16_supported = False
    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False
    
    # 如果开启 AMP，根据硬件支持选择 bfloat16 或 float16
    amp_dtype = torch.bfloat16 if (use_amp and bf16_supported) else torch.float16
    
    # Scaler 仅在使用 float16 时需要；bfloat16 不需要 scaler
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=2.0**10)

    print(f"AMP Config: Enabled={use_amp}, Dtype={amp_dtype}, Scaler={use_scaler}")

    #w
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase = 'train', is_training = True)




    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    #w
    eval_loader = [data_loader_fn(args, phase = 'val', is_training = False)]
    if getattr(args, 'reset_scheduler', False):
        args.scheduler_step_offset = int(logger.global_step)


    evaluator = ModelEvaluator1(
        args.dataset,
        eval_loader,
        args.agg_method,
        args.epochs_per_eval,
        shap_eval_freq=args.shap_eval_freq,
        ablation_eval_freq=args.ablation_eval_freq,
        use_amp=use_amp,
    )
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # 将 table 转换为以 NewPatientID 为键的字典，极大提升查询速度
    # feature_cols is already defined above
    
    # 修复：确保 NewPatientID 唯一，否则 set_index 会报错
    # 如果有重复 ID，drop_duplicates 默认保留第一个
    if not table['NewPatientID'].is_unique:
        print(f"Warning: Duplicate NewPatientID found in metadata (likely due to oversampling in CSV).")
        print(f"         Dropping duplicates ONLY in the metadata lookup table to ensure unique feature mapping.")
        print(f"         Rest assured: This DOES NOT affect the actual training dataset size/oversampling (determined by .pkl file).")
        table = table.drop_duplicates(subset=['NewPatientID'], keep='first')
        
    table_dict = table.set_index('NewPatientID')[feature_cols].to_dict('index')
    default_data = np.zeros(len(feature_cols), dtype=np.float32)

    # -------------------------------------
    # 准备外部验证集 (如果指定了路径)
    # -------------------------------------
    external_loader = None
    if args.external_test_path and os.path.exists(args.external_test_path):
        print(f"Loading external test set from: {args.external_test_path}")
        from data_loader.external_tab_loader import get_external_loader
        
        # 假设 scaler 已经被 fit 过了 (如果 feature_cols 存在)
        scaler_for_ext = scaler if 'scaler' in locals() else None
        
        try:
            external_loader = get_external_loader(
                excel_path=args.external_test_path,
                feature_cols=feature_cols,
                scaler=scaler_for_ext,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            print(f"External test set loaded: {len(external_loader.dataset)} samples.")
            
            # 初始化一个专用的 Evaluator，只做 TabOnly 评估
            # 注意：这里我们不需要 ablation，因为我们只跑 TabOnly 模式
            # 但 ModelEvaluator1.evaluate 内部逻辑较复杂，我们稍后在循环中单独调用
            external_evaluator = ModelEvaluator1(
                "ExternalTab", # 虚拟 dataset 名
                [external_loader],
                args.agg_method,
                epochs_per_eval=1,
                shap_eval_freq=0,
                ablation_eval_freq=0,
                use_amp=use_amp,
            )
            
        except Exception as e:
            print(f"Failed to load external test set: {e}")
            external_loader = None
    # -------------------------------------

    
    # 记录总开始时间
    total_start_time = time.time()
    
    # Train model
    while not logger.is_finished_training():
        _configure_branch_training(model, args)
        logger.start_epoch()

        # --- Performance Profiling ---
        batch_time = util.AverageMeter()
        data_time = util.AverageMeter()
        forward_time = util.AverageMeter()
        backward_time = util.AverageMeter()
        end = time.time()
        # -----------------------------

        with tqdm(
            total=len(train_loader),
            desc=f'Epoch {logger.epoch}',
            unit='batch',
            dynamic_ncols=True,
            mininterval=2.0,
            smoothing=0.0,
        ) as pbar:
            for batch_idx, (img, target_dict) in enumerate(train_loader):
                # Measure data loading time
                data_time.update(time.time() - end)
                
                logger.start_iter()
                
                ###
                ids = [item for item in target_dict['study_num']]
                tab=[]
                
                for pid in ids:
                    if pid in table_dict:
                        row = table_dict[pid]
                        data = np.array([row[c] for c in feature_cols], dtype=np.float32)
                    else:
                        data = default_data
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab = torch.stack(tab)
    
                with torch.set_grad_enabled(True):
                    # 使用非阻塞传输
                    img = img.to(args.device, non_blocking=True)
                    tab = tab.to(args.device, non_blocking=True)
                    # 显式转换为 float，确保损失计算的数值稳定性
                    label = target_dict['is_abnormal'].to(args.device, non_blocking=True).float()

                    if not torch.isfinite(img).all():
                        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    if not torch.isfinite(tab).all():
                        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
                    if not torch.isfinite(label).all():
                        label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    loss_value = float('nan')
                    t_fwd = time.time()
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                        mode_flag = getattr(args, 'forward_mode_flag', 0)
                        if mode_flag != 0:
                            try:
                                out = model.forward(img, tab, mode=mode_flag)
                            except TypeError:
                                out = model.forward(img, tab)
                        else:
                            out = model.forward(img, tab)
                        label_ = label.unsqueeze(1)
                        loss = cls_loss_fn(out, label_).mean()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_time.update(time.time() - t_fwd)
                    loss_value = float(loss.detach().item())

                    if (out is None) or (loss is None) or (not torch.isfinite(out).all()) or (not torch.isfinite(loss)):
                        current_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else float('nan')
                        scale_str = scaler.get_scale() if use_scaler else 'n/a'
                        mem_str = "n/a"
                        if torch.cuda.is_available():
                            mem_str = f"{torch.cuda.memory_allocated()/1024**3:.2f}GiB/{torch.cuda.memory_reserved()/1024**3:.2f}GiB"
                        print(
                            "Warning: non-finite detected; skipping backward. "
                            f"iter={logger.global_step} lr={current_lr} "
                            f"out_finite={bool(torch.isfinite(out).all().item()) if out is not None else False} "
                            f"loss={loss_value} "
                            f"img_shape={tuple(img.shape)} tab_shape={tuple(tab.shape)} "
                            f"img_min={img.min().item():.4g} img_max={img.max().item():.4g} "
                            f"tab_min={tab.min().item():.4g} tab_max={tab.max().item():.4g} "
                            f"scale={scale_str} mem(alloc/res)={mem_str}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
    
                    t_bwd = time.time()
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    backward_time.update(time.time() - t_bwd)

                    if use_scaler:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not torch.isfinite(grad_norm):
                        current_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else float('nan')
                        bad_grad_names = []
                        bad_grad_stats = []
                        try:
                            for name, param in model.named_parameters():
                                if param.grad is None:
                                    continue
                                grad = param.grad
                                if not torch.isfinite(grad).all():
                                    bad_grad_names.append(name)
                                    grad_detached = grad.detach()
                                    bad_grad_stats.append(
                                        (
                                            name,
                                            float(grad_detached.min().item()),
                                            float(grad_detached.max().item()),
                                            str(grad_detached.dtype),
                                        )
                                    )
                                    if len(bad_grad_names) >= 5:
                                        break
                        except Exception:
                            pass
                        scale_str = scaler.get_scale() if use_scaler else 'n/a'
                        mem_str = "n/a"
                        if torch.cuda.is_available():
                            mem_str = f"{torch.cuda.memory_allocated()/1024**3:.2f}GiB/{torch.cuda.memory_reserved()/1024**3:.2f}GiB"
                        print(
                            "Warning: non-finite grad_norm; skipping update. "
                            f"iter={logger.global_step} lr={current_lr} grad_norm={grad_norm.item()} "
                            f"scale={scale_str} mem(alloc/res)={mem_str} "
                            f"loss={loss_value:.6g} "
                            f"out_min={out.min().item():.4g} out_max={out.max().item():.4g} "
                            f"img_min={img.min().item():.4g} img_max={img.max().item():.4g} "
                            f"tab_min={tab.min().item():.4g} tab_max={tab.max().item():.4g} "
                            f"bad_grads={bad_grad_names if len(bad_grad_names) > 0 else 'n/a'} "
                            f"bad_grad_stats={bad_grad_stats if len(bad_grad_stats) > 0 else 'n/a'}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if use_scaler:
                            scaler.update()
                        continue
                    
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
    
    
            
                    ###Log
                    logger.log_iter(img,None,None,loss,optimizer)
    
                logger.end_iter()
                util.step_scheduler(lr_scheduler, global_step=_get_scheduler_step(logger, args))
                # util.step_scheduler(D_lr_scheduler,global_step=logger.global_step)
                
                # Measure total batch time
                batch_time.update(time.time() - end)
                end = time.time()

                if (batch_idx % 10 == 0) or (batch_idx == len(train_loader) - 1):
                    pbar.set_postfix(
                        loss=f"{loss_value:.4f}" if np.isfinite(loss_value) else "nan",
                        data=f"{data_time.avg:.3f}",
                        fwd=f"{forward_time.avg:.3f}",
                        bwd=f"{backward_time.avg:.3f}",
                        tot=f"{batch_time.avg:.3f}"
                    )
                pbar.update(1)

        ###
        # Evaluate on validation set
        print('Validation...')
        # 传入 num_epochs 供 evaluator 判断是否为最后一轮
        metrics, curves, avg_loss = evaluator.evaluate(model, args.device, logger.epoch, num_epochs=args.num_epochs, table=table, train_mode=getattr(args, 'train_mode', 'multimodal'))
        
        # -------------------------------------
        # 执行外部验证 (TabOnly) + 训练集 TabOnly 对比
        # 频率：由 run_multi_experiments.ps1 的 eval_freq 传入 external_eval_freq 控制
        # -------------------------------------
        do_external_eval = bool(external_loader is not None and args.external_eval_freq > 0 and (logger.epoch % args.external_eval_freq == 0 or logger.epoch >= args.num_epochs))
        if do_external_eval:
            print(f'\nRunning External Validation (TabOnly) on {args.external_test_path}...')

            ext_metrics, ext_curves, _ = external_evaluator.evaluate(
                model,
                args.device,
                logger.epoch,
                num_epochs=args.num_epochs,
                table=external_loader.dataset.table_df,
                train_mode='tab_only' # External is always TabOnly
            )

            for k, v in ext_metrics.items():
                metrics[f'External_{k}'] = v

            for k, v in ext_curves.items():
                if isinstance(v, tuple) and len(v) > 0 and len(v[0]) > 0:
                    curves[f'External_{k}'] = v
                elif isinstance(v, np.ndarray):
                    curves[f'External_{k}'] = v
                
                # 打印重点指标
                print(f"External Validation Results: Loss={ext_metrics.get('external_test_loss', -1):.4f}, "
                      f"AUROC={ext_metrics.get('external_test_AUROC', -1):.4f}, "
                      f"Acc={ext_metrics.get('external_test_Accuracy', -1):.4f}")
        # -------------------------------------
        # 训练集 TabOnly（用于和 val/external 做对比），只在外部验证同频率下计算
        # -------------------------------------
        if do_external_eval:
            train_eval_loader = data_loader_fn(args, phase='train', is_training=False)
            train_evaluator = ModelEvaluator1(
                args.dataset,
                [train_eval_loader],
                args.agg_method,
                epochs_per_eval=1,
                shap_eval_freq=0,
                ablation_eval_freq=args.external_eval_freq,
                use_amp=use_amp,
            )
            train_metrics, train_curves, _ = train_evaluator.evaluate(
                model,
                args.device,
                logger.epoch,
                num_epochs=args.num_epochs,
                table=table,
                train_mode=getattr(args, 'train_mode', 'multimodal')
            )
            metrics.update(train_metrics)
            curves.update(train_curves)
        saver.save(logger.epoch,model,optimizer,lr_scheduler,args.device,
                    metric_val=(avg_loss if np.isfinite(avg_loss) else metrics.get('val_loss', avg_loss)))#W metrics.get(args.best_ckpt_metric,None)  avg_loss
        print(metrics)
        logger.end_epoch(metrics,curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)
        #util.step_scheduler(D_lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)
        
    # 训练结束，记录总时长
    total_duration = time.time() - total_start_time
    import datetime
    total_time_str = str(datetime.timedelta(seconds=int(total_duration)))
    print(f"Training Finished. Total duration: {total_time_str}")
    
    # 写入 TensorBoard
    if hasattr(logger, 'writer') and logger.writer is not None:
        logger.writer.add_text('Total_Training_Time', total_time_str, global_step=logger.epoch)
        # 也可以记录为 scalar 方便比较，单位：小时
        logger.writer.add_scalar('Total_Training_Hours', total_duration / 3600.0, global_step=logger.epoch)
        logger.writer.close()

    # 显式清理
    del train_loader
    del eval_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

###
if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()
    args_ = parser.parse_args()

    # Load preprocessed table data
    print(f"Loading metadata from: {args_.data_dir}/G_first_last_nor.csv")
    csv_path = os.path.join(args_.data_dir, 'G_first_last_nor.csv')
    tab = pd.read_csv(csv_path)
    
    # 移除原有的 KNN 和归一化逻辑，因为 preprocess_data.py 已经处理过了
    print("Metadata loaded (already imputed and normalized).")

    # 移除原有的 KNN 和归一化逻辑，因为 preprocess_data.py 已经处理过了
    print("Metadata loaded (already imputed and normalized).")

    ###
    train(args_,tab)
