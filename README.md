adma优化器会产生nan错误，改成取消部分amp就好了

消融实验出现nan咋办？
Running Ablation Study for val (Epoch 20)...
CRITICAL WARNING: Model output contains NaN at batch 3 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 5 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 6 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 13 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 34 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 37 (Mode=img_only)!
CRITICAL WARNING: Model output contains NaN at batch 38 (Mode=img_only)!
CRITICAL: img_only evaluation failed due to NaN or invalid values: Model output contains NaN
