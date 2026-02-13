adma优化器会产生nan错误，改成取消部分amp就好了

消融实验出现nan？
CRITICAL: img_only evaluation failed due to NaN or invalid values: Model output contains NaN
（消融evaluation时没有取消amp
