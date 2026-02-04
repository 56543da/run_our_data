# TensorBoard Background Starter Script

$tensorboard_exe = "C:\ProgramData\miniconda3\envs\mmcafnet\Scripts\tensorboard.exe"
$log_dir = "E:\rerun2\MMCAF-Net-main\logs"

if (Test-Path $tensorboard_exe) {
    Write-Host "Starting TensorBoard in background..." -ForegroundColor Green
    Write-Host "Log Directory: $log_dir" -ForegroundColor Cyan
    
    # Start TensorBoard as a hidden background process
    Start-Process $tensorboard_exe -ArgumentList "--logdir=`"$log_dir`"" -WindowStyle Hidden
    
    Write-Host "TensorBoard is now running in the background." -ForegroundColor Green
    Write-Host "You can access it at http://localhost:6006" -ForegroundColor Yellow
    Write-Host "It will stay running even if you close this SSH session." -ForegroundColor Gray
} else {
    Write-Error "Could not find tensorboard.exe at $tensorboard_exe"
}
