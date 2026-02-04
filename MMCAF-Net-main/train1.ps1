# Windows PowerShell 训练脚本
# 使用 mmcafnet 环境的 Python

# 确保在脚本所在目录下运行
Set-Location $PSScriptRoot

# 自动检测 Python 环境路径
$C_PATH = "C:\conda_envs\mmcafnet\python.exe"
$C_MINICONDA_PATH = "C:\ProgramData\miniconda3\envs\mmcafnet\python.exe"
$E_PATH = "E:\conda_envs\mmcafnet\python.exe"

if (Test-Path $C_PATH) {
    $PYTHON_EXE = $C_PATH
    Write-Host "Using mmcafnet environment from C drive: $C_PATH" -ForegroundColor Green
} elseif (Test-Path $C_MINICONDA_PATH) {
    $PYTHON_EXE = $C_MINICONDA_PATH
    Write-Host "Using mmcafnet environment from C drive (Miniconda): $C_MINICONDA_PATH" -ForegroundColor Green
} elseif (Test-Path $E_PATH) {
    $PYTHON_EXE = $E_PATH
    Write-Host "Using mmcafnet environment from E drive: $E_PATH" -ForegroundColor Green
} else {
    Write-Error "Could not find mmcafnet environment in C or E drive!"
    exit 1
}

# 使用数组方式组织参数
$trainArgs = @(
    "--data_dir=../data"
    "--save_dir=../train_result"
    "--name=ISBI_total"
    "--model=MMCAF_Net"
    "--batch_size=32"
    "--gpu_ids=0"
    "--iters_per_print=32"
    "--iters_per_visual=8000"
    "--learning_rate=1e-4"
    "--lr_decay_step=600000"
    "--lr_scheduler=cosine_warmup"
    "--num_epochs=50"
    "--num_slices=12"
    "--weight_decay=1e-4"
    "--phase=train"
    "--agg_method=max"
    "--best_ckpt_metric=val_loss"
    "--crop_shape=192,192"
    "--cudnn_benchmark=False"
    "--dataset=pe"
    "--do_classify=True"
    "--epochs_per_eval=1"
    "--epochs_per_save=1"
    "--fine_tune=False"
    "--fine_tuning_boundary=classifier"
    "--fine_tuning_lr=1e-2"
    "--include_normals=True"
    "--lr_warmup_steps=10000"
    "--model_depth=50"
    "--num_classes=1"
    "--num_visuals=8"
    "--num_workers=4"
    "--optimizer=adam"
    "--pe_types=['central','segmental']"
    "--resize_shape=192,192"
    "--sgd_dampening=0.9"
    "--sgd_momentum=0.9"
    "--use_pretrained=False"
)

Write-Host "Starting training with the following command:" -ForegroundColor Cyan
Write-Host "$PYTHON_EXE train1.py $($trainArgs -join ' ')" -ForegroundColor Gray

# 执行训练命令
& $PYTHON_EXE train1.py @trainArgs
