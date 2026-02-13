import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ExternalTabDataset(Dataset):
    def __init__(self, excel_path, feature_cols, label_col='STAS', img_shape=(1, 192, 192), scaler=None):
        """
        Args:
            excel_path (str): 外部验证集 Excel 文件路径
            feature_cols (list): 需要提取的特征列名列表
            label_col (str): 标签列名 (如 'STAS')
            img_shape (tuple): 伪造图像的形状 (C, H, W) 或 (D, H, W)
            scaler (sklearn.preprocessing.StandardScaler): 预训练好的归一化器 (可选)
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.img_shape = img_shape
        
        # 1. 读取数据
        try:
            self.df = pd.read_excel(excel_path)
        except Exception as e:
            raise ValueError(f"Failed to read external test set from {excel_path}: {e}")
            
        # 2. 检查列是否存在
        missing_feats = [c for c in feature_cols if c not in self.df.columns]
        if missing_feats:
            print(f"WARNING: External dataset missing features: {missing_feats}. Filling with 0.")
            for c in missing_feats:
                self.df[c] = 0.0
                
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found in external dataset.")
            
        # 3. 处理数据 (缺失值填充 + 归一化)
        # 提取特征矩阵
        self.X = self.df[feature_cols].copy()
        # 简单填充缺失值
        self.X = self.X.fillna(0.0)
        
        # 归一化
        if scaler is not None:
            # 使用训练集的 scaler
            self.X_scaled = scaler.transform(self.X)
        else:
            # 如果没有提供 scaler，则不进行归一化 (或使用自身统计量，但这在迁移学习中不推荐)
            print("WARNING: No scaler provided for external test set. Using raw values (may degrade performance).")
            self.X_scaled = self.X.values
            
        # 提取标签并转换为 0/1 (假设 'STAS' 已经是 0/1 或需要映射)
        # 这里假设 Excel 中标签已经是数值，或者需要简单映射
        self.y = self.df[label_col].values
        
        # 确保标签是数值型
        if self.y.dtype == object:
            # 尝试转换，例如 '是'/'否' -> 1/0
            self.y = pd.to_numeric(self.y, errors='coerce')
            self.y = np.nan_to_num(self.y, nan=0.0)  # 转换失败默认为 0

        self.y = np.asarray(self.y, dtype=np.float32)

        self.df = self.df.copy()
        if 'NewPatientID' not in self.df.columns:
            self.df['NewPatientID'] = [f"EXT_{i}" for i in range(len(self.df))]
        self.table_df = self.df[['NewPatientID'] + list(feature_cols)].copy()
        self.table_df[label_col] = self.y
             
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 2. 获取标签
        label = float(self.y[idx])
        
        # 3. 构造伪造图像 (全零)
        # MMCAF-Net 期望输入是 (C, D, H, W) 或 (C, H, W)
        # 根据 train1.py 中的 img 处理，通常是 (1, D, H, W) 或 (1, H, W)
        # 这里我们生成全零张量，形状由 img_shape 决定
        # 注意：ModelEvaluator1 会再次 unsqueeze batch 维度
        fake_img = torch.zeros(self.img_shape, dtype=torch.float32)
        
        # 4. 构造 targets_dict (为了兼容 ModelEvaluator1)
        targets_dict = {
            'is_abnormal': torch.tensor(label, dtype=torch.float32),
            'study_num': f"EXT_{idx}",
            'series_idx': torch.tensor(idx, dtype=torch.long),
        }
        
        return fake_img, targets_dict

def get_external_loader(excel_path, feature_cols, scaler, batch_size=16, num_workers=4):
    dataset = ExternalTabDataset(
        excel_path=excel_path,
        feature_cols=feature_cols,
        scaler=scaler,
        img_shape=(1, 12, 192, 192) # 假设 3D 输入 (C, D, H, W)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 猴子补丁：为 loader 添加 phase 属性，兼容 evaluator
    loader.phase = 'external_test'
    loader.get_series_label = lambda idx: float(dataset.y[int(idx)])
    
    return loader
