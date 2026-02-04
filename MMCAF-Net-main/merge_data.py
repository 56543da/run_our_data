
import pandas as pd
import numpy as np
import os

# 自动识别路径：假设脚本位于 MMCAF-Net-main 目录下
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # e:\rerun2
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

g_path = os.path.join(DATA_DIR, 'G_first_last_nor.csv')
meta_path = os.path.join(DATA_DIR, 'metadata.csv')

print(f"Looking for files in: {DATA_DIR}")

try:
    g_df = pd.read_csv(g_path)
    meta_df = pd.read_csv(meta_path)
    print(f"Loaded {len(g_df)} rows from G_first_last_nor.csv")
    print(f"Loaded {len(meta_df)} rows from metadata.csv")
except Exception as e:
    print(f"Error loading files: {e}")
    print(f"Please ensure 'G_first_last_nor.csv' and 'metadata.csv' exist in {DATA_DIR}")
    exit(1)

desired_feature_cols = ['实性成分大小', '毛刺征', '支气管异常征', '胸膜凹陷征', 'CEA']

# Create a dictionary for fast lookup
meta_dict = {}
for _, row in meta_df.iterrows():
    pid = str(row['NewPatientID']).strip()
    clin_data = {}
    for c in desired_feature_cols:
        v = row.get(c, 0.0)
        try:
            clin_data[c] = float(v)
        except Exception:
            clin_data[c] = 0.0
    meta_dict[pid] = clin_data

# Update G dataframe
new_rows = []
for _, row in g_df.iterrows():
    pid = str(row['NewPatientID']).strip()
    clin_data = meta_dict.get(pid, {c: 0.0 for c in desired_feature_cols})
    
    # Create new row dict
    new_row = row.to_dict()
    new_row.update(clin_data)
    new_rows.append(new_row)

df_out = pd.DataFrame(new_rows)

# Reorder columns
meta_cols = ['NewPatientID'] + desired_feature_cols
tail_cols = ['label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
cols = [c for c in meta_cols if c in df_out.columns] + [c for c in tail_cols if c in df_out.columns]
df_out = df_out[cols]

# Save
df_out.to_csv(g_path, index=False)
print("Successfully merged clinical data.")
print("First few rows:")
print(df_out.head())
