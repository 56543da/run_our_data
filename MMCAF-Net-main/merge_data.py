
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

# Process metadata
sex_map = {'M': 1, 'F': 0}
t_map = {
    'is': 0, '1': 1, '1a': 1, '1b': 2, '1c': 3, 
    '2': 4, '2a': 4, '2b': 5, '3': 6, '4': 7
}
m_map = {
    '0': 0, '1': 1, '1a': 1, '1b': 2, '1c': 3, 
    '2': 4, '3': 5
}

# Create a dictionary for fast lookup
meta_dict = {}
for _, row in meta_df.iterrows():
    pid = str(row['NewPatientID']).strip()
    
    sex = sex_map.get(row['Sex'], 0)
    try: age = float(row['Age'])
    except: age = 0.0
    try: weight = float(row['weight (kg)'])
    except: weight = 0.0
    t_stage = t_map.get(str(row['T-Stage']).strip(), 0)
    try: n_stage = int(row['N-Stage'])
    except: n_stage = 0
    
    m_col = 'Ｍ-Stage' if 'Ｍ-Stage' in row else 'M-Stage'
    m_val = str(row.get(m_col, '0')).strip()
    m_stage = m_map.get(m_val, 0)
    
    try: smoking = int(row['Smoking History'])
    except: smoking = 0
    
    meta_dict[pid] = {
        'Sex': sex, 'Age': age, 'Weight': weight,
        'T-Stage': t_stage, 'N-Stage': n_stage, 'M-Stage': m_stage,
        'Smoking': smoking
    }

# Update G dataframe
new_rows = []
for _, row in g_df.iterrows():
    pid = str(row['NewPatientID']).strip()
    clin_data = meta_dict.get(pid, {
        'Sex': 0, 'Age': 0, 'Weight': 0,
        'T-Stage': 0, 'N-Stage': 0, 'M-Stage': 0,
        'Smoking': 0
    })
    
    # Create new row dict
    new_row = row.to_dict()
    new_row.update(clin_data)
    new_rows.append(new_row)

df_out = pd.DataFrame(new_rows)

# Reorder columns
cols = ['NewPatientID', 'Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking', 
        'label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
df_out = df_out[cols]

# Save
df_out.to_csv(g_path, index=False)
print("Successfully merged clinical data.")
print("First few rows:")
print(df_out.head())
