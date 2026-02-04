
import models
import torch
import torch.nn as nn
import util
import warnings
import os

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
from tqdm import tqdm
from sklearn.impute import KNNImputer

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
    #w
    count = 0
    much = 50

    if args.ckpt_path and not args.use_pretrained:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
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


    evaluator = ModelEvaluator1(args.dataset, eval_loader, args.agg_method, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # 将 table 转换为以 NewPatientID 为键的字典，极大提升查询速度
    feature_cols = ['Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking']
    
    # 修复：确保 NewPatientID 唯一，否则 set_index 会报错
    # 如果有重复 ID，drop_duplicates 默认保留第一个
    if not table['NewPatientID'].is_unique:
        print(f"Warning: Duplicate NewPatientID found in metadata. Dropping duplicates...")
        table = table.drop_duplicates(subset=['NewPatientID'], keep='first')
        
    table_dict = table.set_index('NewPatientID')[feature_cols].to_dict('index')
    default_data = np.zeros(7, dtype=np.float32)

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        with tqdm(total=len(train_loader), desc=f'Epoch {logger.epoch}', unit='batch') as pbar:
            for img, target_dict in train_loader:
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
                    with torch.cuda.amp.autocast():
                        out = model.forward(img, tab)
                        label = label.unsqueeze(1)
                        cls_loss = cls_loss_fn(out, label).mean()
                        loss = cls_loss
    
                    # 三步走 (配合 AMP)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
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
                
                # 更新进度条
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        ###
        # Evaluate on validation set
        print('Validation...')
        # 传入 num_epochs 供 evaluator 判断是否为最后一轮
        metrics,curves,avg_loss=evaluator.evaluate(model,args.device,logger.epoch, num_epochs=args.num_epochs, table=table)
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
