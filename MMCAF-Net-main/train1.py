
import models
import torch
import torch.nn as nn
import util
import warnings
import os

import random
import numpy as np

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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
    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer, args)

    if args.ckpt_path and not args.use_pretrained and not args.fine_tune:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
        # ModelSaver.load_optimizer(args.ckpt_path, optimizer, D_lr_scheduler)

    # Get logger, evaluator, saver
    # 恢复标准 BCE Loss，严格遵循论文设定
    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    
    # 初始化 AMP 梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    #w
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase = 'train', is_training = True)




    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    #w
    eval_loader = [data_loader_fn(args, phase = 'val', is_training = False)]


    evaluator = ModelEvaluator1(args.dataset, eval_loader, args.agg_method, args.epochs_per_eval, shap_eval_freq=args.shap_eval_freq, ablation_eval_freq=args.ablation_eval_freq)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # 将 table 转换为以 NewPatientID 为键的字典，极大提升查询速度
    # feature_cols is already defined above
    
    # 修复：确保 NewPatientID 唯一，否则 set_index 会报错
    # 如果有重复 ID，drop_duplicates 默认保留第一个
    if not table['NewPatientID'].is_unique:
        print(f"Warning: Duplicate NewPatientID found in metadata. Dropping duplicates...")
        table = table.drop_duplicates(subset=['NewPatientID'], keep='first')
        
    table_dict = table.set_index('NewPatientID')[feature_cols].to_dict('index')
    default_data = np.zeros(len(feature_cols), dtype=np.float32)

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        # --- Performance Profiling ---
        batch_time = util.AverageMeter()
        data_time = util.AverageMeter()
        forward_time = util.AverageMeter()
        backward_time = util.AverageMeter()
        end = time.time()
        # -----------------------------

        with tqdm(total=len(train_loader), desc=f'Epoch {logger.epoch}', unit='batch') as pbar:
            for img, target_dict in train_loader:
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
                    
                    # 使用自动混合精度 (AMP) 加速
                    t_fwd = time.time()
                    with torch.cuda.amp.autocast():
                        out = model.forward(img, tab)
                        label = label.unsqueeze(1)
                        cls_loss = cls_loss_fn(out, label).mean()
                        loss = cls_loss
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    forward_time.update(time.time() - t_fwd)
    
                    # 三步走 (配合 AMP)
                    t_bwd = time.time()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    backward_time.update(time.time() - t_bwd)
                    
                    # 检查梯度是否出现 NaN/Inf (由混合精度引起)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪防止抖动
                    
                    # 只有在 loss 不是 NaN 的情况下才更新权重
                    if not torch.isnan(loss):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        print(f"Warning: NaN loss detected at iter {logger.global_step}, skipping update.")
                        scaler.update() # 仍然需要 update 来调整 scaler 的 scale
    
    
            
                    ###Log
                    logger.log_iter(img,None,None,loss,optimizer)
    
                logger.end_iter()
                util.step_scheduler(lr_scheduler,global_step=logger.global_step)
                # util.step_scheduler(D_lr_scheduler,global_step=logger.global_step)
                
                # Measure total batch time
                batch_time.update(time.time() - end)
                end = time.time()

                # 更新进度条
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
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
        metrics,curves,avg_loss=evaluator.evaluate(model,args.device,logger.epoch, num_epochs=args.num_epochs, table=table)
        if logger.epoch >= args.num_epochs:
            train_eval_loader = data_loader_fn(args, phase='train', is_training=False)
            train_evaluator = ModelEvaluator1(
                args.dataset,
                [train_eval_loader],
                args.agg_method,
                args.epochs_per_eval,
                shap_eval_freq=0,
                ablation_eval_freq=0
            )
            train_metrics, train_curves, _ = train_evaluator.evaluate(
                model,
                args.device,
                logger.epoch,
                num_epochs=args.num_epochs,
                table=table
            )
            metrics.update(train_metrics)
            curves.update(train_curves)
        saver.save(logger.epoch,model,optimizer,lr_scheduler,args.device,
                    metric_val=avg_loss)#W metrics.get(args.best_ckpt_metric,None)  avg_loss
        print(metrics)
        logger.end_epoch(metrics,curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)
        #util.step_scheduler(D_lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)

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
