import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats
import os

# 配置
EXCEL_PATH = r'e:\run_our_data\data\原始STAS_data.xlsx'
OUTPUT_DIR = r'e:\run_our_data\feature_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体 (尝试常见中文字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_features():
    print(f"Loading data from {EXCEL_PATH}...")
    df = pd.read_excel(EXCEL_PATH)
    
    # 1. 预处理：识别特征列和标签列
    # 假设标签列名为 'STAS'，如果不是请修改
    label_col = 'STAS'
    if label_col not in df.columns:
        print(f"Error: Label column '{label_col}' not found!")
        return

    # 排除非数值列（如姓名、ID、病理描述等）
    exclude_cols = ['Name', '姓名', 'PatientName', 'PatientsName', '病理号', 'ID', label_col, '是否转移', '病理结果']
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # 额外加入一些可能的分类变量但被存为数值的（如果有的话）
    
    print(f"Analyzing {len(feature_cols)} features against label '{label_col}'...")
    
    results = []
    
    for feat in feature_cols:
        # 去除空值进行计算
        valid_data = df[[feat, label_col]].dropna()
        if len(valid_data) < 10: continue
        
        X = valid_data[feat]
        y = valid_data[label_col]
        
        # 1. Point-Biserial Correlation (对于连续变量 vs 二分类)
        corr, p_val = stats.pointbiserialr(X, y)
        
        # 2. 单变量 AUC
        # 注意：如果特征与标签负相关，AUC可能会 < 0.5，需要反转
        try:
            auc = roc_auc_score(y, X)
            if auc < 0.5: auc = 1.0 - auc # 处理负相关特征
        except:
            auc = 0.5
            
        # 3. T-test / Mann-Whitney U test
        group0 = X[y == 0]
        group1 = X[y == 1]
        try:
            _, u_p_val = stats.mannwhitneyu(group0, group1)
        except:
            u_p_val = 1.0
            
        results.append({
            'Feature': feat,
            'Correlation': corr,
            'Abs_Correlation': abs(corr),
            'AUC': auc,
            'P_Value': p_val,
            'U_Test_P': u_p_val
        })

    # 转换为 DataFrame 并排序
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('AUC', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 10 SUSPICIOUS FEATURES (Ranked by AUC)")
    print("="*50)
    print(res_df[['Feature', 'AUC', 'Correlation', 'P_Value']].head(10).to_string(index=False))
    
    # 保存结果
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_leakage_analysis.csv'), index=False)
    
    # --- 绘图 ---
    
    # 1. Top 10 特征的相关性条形图
    plt.figure(figsize=(12, 6))
    top_10 = res_df.head(10)
    sns.barplot(x='AUC', y='Feature', data=top_10, palette='viridis')
    plt.axvline(0.5, color='r', linestyle='--', label='Random Guess')
    plt.axvline(0.8, color='orange', linestyle='--', label='Strong Predictor')
    plt.axvline(0.95, color='red', linestyle='--', label='Potential Leakage')
    plt.title('Top 10 Features by Single-Variable AUC')
    plt.xlabel('AUC (predictive power)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_features_auc.png'))
    plt.close()
    
    # 2. Top 6 特征的箱线图 (分布差异)
    top_6_feats = res_df['Feature'].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_6_feats):
        sns.boxplot(x=label_col, y=feat, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f"{feat} (AUC={res_df[res_df['Feature']==feat]['AUC'].values[0]:.3f})")
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_features_boxplot.png'))
    plt.close()
    
    print(f"\nAnalysis complete! Report and charts saved to: {OUTPUT_DIR}")
    print("Look for features with AUC > 0.9 or Correlation > 0.7 - these are likely leakage sources.")

if __name__ == "__main__":
    analyze_features()
