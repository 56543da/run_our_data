# MMCAF-Net 私有数据集生成指南

本文档介绍如何使用 `MMCAF-Net-main` 目录下的核心脚本来处理您的私有数据（NIfTI 影像 + Excel 表格），生成模型训练所需的 HDF5、PKL 和 CSV 文件。

## 1. 脚本功能概览

| 脚本名称 | 核心功能 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| [preprocess_data.py](file:///e:/run_our_data/MMCAF-Net-main/preprocess_data.py) | **元数据提取与平衡**：扫描 NIfTI 文件夹（0/1 结构），提取 BBox，关联 Excel 临床信息，并进行过采样平衡。 | `Lung.nii`, `ROI.nii`, `原始STAS_data.xlsx` | `G_first_last_nor.csv` |
| [hdf5_ours.py](file:///e:/run_our_data/MMCAF-Net-main/hdf5_ours.py) | **影像数据打包**：将所有 `Lung.nii` 影像文件高效打包进一个 HDF5 文件。已优化内存占用 (float32)。 | `Lung.nii` | `data.hdf5` |
| [pkl_read.py](file:///e:/run_our_data/MMCAF-Net-main/pkl_read.py) | **生成序列索引**：将生成的 CSV 转换为模型直接读取的 Pickle 索引对象。 | `G_first_last_nor.csv` | `series_list.pkl` |

---

## 2. 数据准备流程

请按以下顺序执行脚本。推荐使用 **C 盘** 的环境路径：

### 第一步：生成元数据 CSV (Preprocess)
此脚本会计算病灶的切片范围和平均 BBox，并从 Excel 中提取特征（性别、年龄、BMI/体重、吸烟史）。它会自动处理缺失值并针对 STAS 类别进行平衡。
```powershell
C:\conda_envs\mmcafnet\python.exe e:\run_our_data\MMCAF-Net-main\preprocess_data.py `
    --dcm_root "E:\run_our_data\data\Lung" `
    --meta_file "E:\run_our_data\data\原始STAS_data.xlsx" `
    --output_csv "E:\run_our_data\data\G_first_last_nor.csv"
```

### 第二步：生成 HDF5 影像包 (HDF5)
此脚本将 3D 影像打包。如果内存不足，请适当减少 `--threads` 数值。
```powershell
C:\conda_envs\mmcafnet\python.exe e:\run_our_data\MMCAF-Net-main\hdf5_ours.py `
    --dcm_root "E:\run_our_data\data\Lung" `
    --output_h5 "E:\run_our_data\data\data.hdf5" `
    --threads 8
```

### 第三步：生成序列索引 (Pickle)
将 CSV 转换为模型 Dataset 类识别的 `.pkl` 文件。
```powershell
C:\conda_envs\mmcafnet\python.exe e:\run_our_data\MMCAF-Net-main\pkl_read.py `
    --input_csv "E:\run_our_data\data\G_first_last_nor.csv" `
    --output_pkl "E:\run_our_data\data\series_list.pkl"
```

---

## 3. 常见操作与故障排除

### 内存溢出 (OOM) 报错
如果在运行 `hdf5_ours.py` 时出现 `Unable to allocate ... MiB` 错误：
1. 脚本已默认使用 `float32` 以节省一半内存。
2. 尝试将 `--threads 8` 降低为 `--threads 4` 或 `--threads 2`。

### 数据路径规范
确保您的 `data` 目录结构如下：
```text
E:\run_our_data\data\
├── Lung\
│   ├── 0\              # 阴性样本文件夹
│   │   └── Patient_A\
│   │       ├── Lung.nii
│   │       └── ROI.nii
│   └── 1\              # 阳性样本文件夹
│       └── Patient_B\
│           ├── Lung.nii
│           └── ROI.nii
├── 原始STAS_data.xlsx  # 临床信息表
├── G_first_last_nor.csv # 自动生成
├── data.hdf5           # 自动生成
└── series_list.pkl     # 自动生成
```

---
*文档更新日期：2026-01-26*
