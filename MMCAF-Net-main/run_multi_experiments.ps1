# Windows PowerShell Multi-Experiment Automation Script

# 自动处理互斥锁：如果发现已有进程在运行，自动终止它
$LockFile = Join-Path $PSScriptRoot "experiment.lock"
if (Test-Path $LockFile) {
    $OldProcessId = Get-Content $LockFile
    if (Get-Process -Id $OldProcessId -ErrorAction SilentlyContinue) {
        Write-Host "Detecting an existing experiment (PID: $OldProcessId). Automatically stopping it..." -ForegroundColor Yellow
        Stop-Process -Id $OldProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2 # 等待进程彻底释放资源
    }
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}

if ($null -eq $env:RUNNING_IN_BACKGROUND) {
    $env:RUNNING_IN_BACKGROUND = "True"
    $logFile = Join-Path $PSScriptRoot "experiments_progress.log"
    # 清空旧日志
    if (Test-Path $logFile) { Remove-Item $logFile -ErrorAction SilentlyContinue }
    
    # 启动后台进程，直接重定向输出到文件，这比 Start-Transcript 反应快得多
    # -WindowStyle Hidden 确保它不会弹出一个新窗口依赖当前的 SSH 会话
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$PSCommandPath *>> '$logFile'`"" -WindowStyle Hidden
    
    Write-Host "Experiments started in background (Unbuffered)!" -ForegroundColor Green
    Write-Host "Streaming logs (Press Ctrl+C to stop viewing, background tasks will continue)..." -ForegroundColor Yellow
    
    # 简单的日志流式读取
    while (-not (Test-Path $logFile)) { Start-Sleep -Milliseconds 500 }
    Get-Content $logFile -Wait
    exit
}

# 确保在脚本所在目录下运行
Set-Location $PSScriptRoot
# 强制 Python 不使用输出缓存
$env:PYTHONUNBUFFERED = "1"
# 优化显存碎片处理
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"

# 记录当前进程 ID 到锁文件
$PID | Out-File $LockFile

$logFile = Join-Path $PSScriptRoot "experiments_progress.log"
if (Test-Path $logFile) { Remove-Item $logFile -ErrorAction SilentlyContinue }

try {
    # Detect Python Environment
    $C_PATH = "C:\conda_envs\mmcafnet\python.exe"
    $C_MINICONDA_PATH = "C:\ProgramData\miniconda3\envs\mmcafnet\python.exe"
    $E_PATH = "E:\conda_envs\mmcafnet\python.exe"

    if (Test-Path $C_PATH) {
        $PYTHON_EXE = $C_PATH
    } elseif (Test-Path $C_MINICONDA_PATH) {
        $PYTHON_EXE = $C_MINICONDA_PATH
    } elseif (Test-Path $E_PATH) {
        $PYTHON_EXE = $E_PATH
    } else {
        Write-Error "Could not find mmcafnet environment!"
        exit 1
    }

    # Define Experiment List
    $experiments = @(
        # 示例：图像单模态训练，设定独立 LR。注意：resume_optim=$false 才能使新 LR 生效
        # 推荐 LR：1e-4 (收敛稍快) 或 5e-5 (更稳健)
        @{ name = "Exp1_adam_B32"; lr = "1e-4"; opt = "adam"; batch = "32"; epochs = "150"; eval_freq = "5"; resume = $true; train_mode = "img_only"; resume_optim = $false }
        
        # 示例：表格单模态训练
        # @{ name = "Exp2_TabOnly"; lr = "1e-3"; opt = "adam"; batch = "32"; epochs = "150"; eval_freq = "5"; resume = $true; bbox = $false; amp = $False; train_mode = "tab_only"; resume_optim = $false }
    )

    foreach ($exp in $experiments) {
        if (-not $exp.ContainsKey('train_mode')) { $exp.train_mode = "multimodal" }
        if (-not $exp.ContainsKey('bbox')) { $exp.bbox = $false }
        if (-not $exp.ContainsKey('amp')) { $exp.amp = $false }
        $resumeOptim = $true
        if ($exp.ContainsKey('resume_optim')) {
            $resumeOptim = [bool]$exp.resume_optim
        } else {
            $resumeOptim = ($exp.train_mode -eq "multimodal")
        }
        
        # 自动逻辑：如果不恢复优化器状态，通常意味着开启新的训练阶段，
        # 此时必须重置调度器步数，否则继承的 global_step 会导致 scheduler 计算出极小的 lr
        $resetScheduler = (-not $resumeOptim)
        if ($exp.ContainsKey('reset_sched')) { $resetScheduler = [bool]$exp.reset_sched }
        
        $schedulerStartStep = 0
        if ($exp.ContainsKey('sched_start')) { $schedulerStartStep = [int]$exp.sched_start }
        $overrideLr = -1.0
        if ($exp.ContainsKey('override_lr')) { $overrideLr = [double]$exp.override_lr }
        # 自动构建外部测试集路径 (相对于 ../data)
        $extTestPath = Join-Path "../data" "外部测试集.xlsx"
        
        Write-Host "`n" + ("="*60) -ForegroundColor Cyan
        Write-Host "Starting Experiment: $($exp.name)" -ForegroundColor Cyan
        Write-Host "Config: Optimizer=$($exp.opt), LR=$($exp.lr), BatchSize=$($exp.batch), Epochs=$($exp.epochs), AMP=$($exp.amp)" -ForegroundColor Cyan
        Write-Host "Eval Config: Analysis Freq=$($exp.eval_freq), Ext Test=$extTestPath" -ForegroundColor Cyan
        Write-Host ("="*60) -ForegroundColor Cyan
        
        # 自动断点续训逻辑
        $ckptArg = ""
        if ($exp.resume -eq $true) {
            # 查找该实验目录下最新的实验文件夹 (带时间戳)
            # 例如: Exp1_adam_B32_20260131_105525
            $baseExpDir = Join-Path "../train_result" ($exp.name + "*")
            
            # 可能会匹配到多个时间戳文件夹，按最后写入时间排序，取最新的一个
            $latestExpDir = Get-ChildItem -Path $baseExpDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

            if ($latestExpDir) {
                 $ckptDir = Join-Path $latestExpDir.FullName "ckpt"
                 # 兼容性检查：如果 ckpt 目录不存在，尝试直接在实验根目录下找 .pth.tar 文件 (如截图所示)
                 if (-not (Test-Path $ckptDir)) {
                     $ckptDir = $latestExpDir.FullName
                 }

                 if (Test-Path $ckptDir) {
                     # 1. 尝试找 last.pth.tar
                     $lastCkpt = Join-Path $ckptDir "last.pth.tar"
                     if (Test-Path $lastCkpt) {
                         $ckptArg = "--ckpt_path=$lastCkpt"
                         Write-Host "Resuming from last checkpoint: $lastCkpt" -ForegroundColor Yellow
                     } else {
                         # 2. 如果没有 last，找最新的 epoch_xx.pth.tar
                         # 获取所有 .pth.tar 文件，排除 best.pth.tar (通常不用于 resume)
                         $allCkpts = Get-ChildItem -Path $ckptDir -Filter "*.pth.tar" | 
                                     Where-Object { $_.Name -ne "best.pth.tar" } | 
                                     Sort-Object LastWriteTime -Descending
                         
                         if ($allCkpts) {
                             # 遍历查找第一个可用的文件
                             foreach ($ckpt in $allCkpts) {
                                 # 这里可以加入额外的完整性检查逻辑(如果需要)，目前假设文件存在即可
                                 # 如果需要更复杂的校验(如文件大小不为0)，可以在这里加 if ($ckpt.Length -gt 0)
                                 
                                 $ckptArg = "--ckpt_path=$($ckpt.FullName)"
                                 Write-Host "Resuming from latest valid checkpoint: $($ckpt.FullName)" -ForegroundColor Yellow
                                 break # 找到最新的一个就退出循环
                             }
                         } else {
                             # 3. 如果连 epoch_xx 都没有，最后尝试 best.pth.tar
                             $bestCkpt = Join-Path $ckptDir "best.pth.tar"
                             if (Test-Path $bestCkpt) {
                                 $ckptArg = "--ckpt_path=$bestCkpt"
                                 Write-Host "No epoch checkpoints found. Resuming from best checkpoint: $bestCkpt" -ForegroundColor Yellow
                             } else {
                                 Write-Host "Resume requested but no valid checkpoints found in $ckptDir." -ForegroundColor Yellow
                             }
                         }
                     }
                 } else {
                     Write-Host "Resume requested but checkpoint directory not found ($ckptDir)." -ForegroundColor Yellow
                 }
            } else {
                 Write-Host "Resume requested but no previous experiment folder found for $($exp.name). Starting fresh." -ForegroundColor Yellow
            }
        }

        # 动态设置 resize_shape
        # 如果启用 bbox 裁剪，需要更大的 resize 尺寸 (256) 以便有裁剪空间 (256->192)
        # 如果不启用 (bbox=$false)，保持 resize=192，这样 resize=crop，即全图输入，不进行随机裁剪
        $resizeShape = "192,192"
        if ($exp.bbox -eq $true) {
            $resizeShape = "256,256"
        }

        $trainArgs = @(
            "--data_dir=../data"
            "--save_dir=../train_result"
            "--name=$($exp.name)"
            "--model=MMCAF_Net"
            "--batch_size=$($exp.batch)"
            "--gpu_ids=0"
            "--iters_per_print=$($exp.batch)"
            "--iters_per_visual=8000"
            "--learning_rate=$($exp.lr)"
            "--lr_decay_step=600000"
            "--lr_scheduler=cosine_warmup"
            "--num_epochs=$($exp.epochs)"
            "--num_slices=12"
            "--weight_decay=1e-4"
            "--phase=train"
            "--agg_method=max"
            "--best_ckpt_metric=val_loss"
            "--crop_shape=192,192"
            "--cudnn_benchmark=True"
            "--dataset=pe"
            "--do_classify=True"
            "--epochs_per_eval=1"
            "--epochs_per_save=1"
            "--shap_eval_freq=$($exp.eval_freq)"
            "--ablation_eval_freq=$($exp.eval_freq)"
            "--fine_tune=False"
            "--fine_tuning_boundary=classifier"
            "--fine_tuning_lr=1e-2"
            "--include_normals=True"
            "--lr_warmup_steps=10000"
            "--model_depth=50"
            "--num_classes=1"
            "--num_visuals=8"
            "--num_workers=8"
            "--optimizer=$($exp.opt)"
            "--pe_types=['central','segmental']"#无效，别慌
            "--resize_shape=$resizeShape"
            "--sgd_dampening=0.9"
            "--sgd_momentum=0.9"
            "--use_pretrained=False"
            "--use_bbox_crop=$($exp.bbox)"
            "--use_amp=$($exp.amp)"
            "--external_test_path=$extTestPath"
            "--external_eval_freq=$($exp.eval_freq)"
            "--train_mode=$($exp.train_mode)"
            "--resume_optimizer=$resumeOptim"
            "--reset_scheduler=$resetScheduler"
            "--scheduler_start_step=$schedulerStartStep"
            "--override_lr=$overrideLr"
        )

        # 执行训练
        if ($ckptArg) {
            & $PYTHON_EXE train1.py @trainArgs $ckptArg
        } else {
            & $PYTHON_EXE train1.py @trainArgs
        }

        Write-Host "Experiment $($exp.name) finished.`n" -ForegroundColor Green
        Start-Sleep -Seconds 10
    }
} finally {
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
}

Write-Host "All experiments completed!" -ForegroundColor Magenta
