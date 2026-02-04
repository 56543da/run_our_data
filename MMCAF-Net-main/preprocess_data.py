import os
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
import re
import nibabel as nib
from sklearn.impute import KNNImputer
import concurrent.futures

# 自动识别路径: 基于当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'Lung', 'lung(train)')
DEFAULT_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'Lung', 'lung(val)')
DEFAULT_METADATA_FILE = os.path.join(PROJECT_ROOT, 'data', '原始STAS_data.xlsx')
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'G_first_last_nor.csv')

# 设置随机种子以保证可复现性
random.seed(42)

# Feature Configuration
# Continuous features to be normalized
CNT = ['实性成分大小', 'CEA']
CONTINUOUS_FEATURES = CNT

# Categorical/Ordinal features (no normalization, just encoding)
CAT = ['毛刺征', '支气管异常征', '胸膜凹陷征']
CATEGORICAL_FEATURES = CAT

# Mapping from Excel headers to Internal names (Fuzzy matching will be used)
COLUMN_MAPPING = {
    '实性成分大小': '实性成分大小',
    '毛刺征': '毛刺征',
    '支气管异常征': '支气管异常征',
    '胸膜凹陷征': '胸膜凹陷征',
    'CEA': 'CEA',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data metadata (Explicit Train/Val Split)')
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DIR, help='Directory of Training NIfTI files')
    parser.add_argument('--val_dir', type=str, default=DEFAULT_VAL_DIR, help='Directory of Validation NIfTI files')
    parser.add_argument('--meta_file', type=str, default=DEFAULT_METADATA_FILE, help='Path to clinical metadata Excel')
    parser.add_argument('--output_csv', type=str, default=DEFAULT_OUTPUT_CSV, help='Output path for the metadata CSV')
    parser.add_argument('--demo', action='store_true', help='Process partial dataset for demo')
    return parser.parse_args()

def process_patient_wrapper(args):
    """Wrapper for parallel processing"""
    # NIfTI mode
    pid, folder, label, _, clin_data, orig_idx = args
    try:
        success, meta = process_patient_nifti(pid, folder, label, {})
        return pid, success, meta, clin_data, orig_idx, label
    except Exception as e:
        return pid, False, None, clin_data, orig_idx, label

def load_and_process_metadata_excel(meta_path, sheet_name=0):
    """读取并处理临床表格数据 (Excel版)"""
    if not os.path.exists(meta_path):
        print(f"Warning: Metadata file not found at {meta_path}")
        return [], []

    try:
        if meta_path.endswith('.csv'):
             df = pd.read_csv(meta_path)
        else:
             print(f"Reading sheet: {sheet_name}")
             df = pd.read_excel(meta_path, sheet_name=sheet_name)
        # Normalize columns: strip whitespace and remove internal spaces for matching
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return [], []
    
    # Identify Name column for ID extraction
    name_col = None
    possible_name_cols = ['Name', '姓名', 'PatientName', 'PatientsName', 'ID']
    for col in df.columns:
        if any(x.lower() == str(col).lower() for x in possible_name_cols):
             name_col = col
        elif any(x in str(col) for x in possible_name_cols):
             if not name_col: name_col = col
             
    if not name_col:
        print("Warning: Could not find Name column in Excel.")
        return [], []

    # Create a reverse mapping for robust column selection
    # Remove spaces from excel headers for comparison
    clean_columns = {str(c).replace(' ', ''): c for c in df.columns}
    
    # 1. Extract raw data
    processed_rows = []
    metadata_list = [] # List of (pid, orig_idx)
    
    target_columns = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    
    print(f"Extracting {len(target_columns)} clinical features...")
    
    for idx, row in df.iterrows():
        # Extract ID
        raw_name = str(row[name_col]).strip()
        # Ensure we catch IDs like P00123456
        id_match = re.search(r'(P\d+)', raw_name)
        if not id_match:
            pid = raw_name
        else:
            pid = id_match.group(1)
            
        row_data = {}
        for excel_header, internal_name in COLUMN_MAPPING.items():
            clean_header = excel_header.replace(' ', '')
            real_col_name = None
            if clean_header in clean_columns:
                real_col_name = clean_columns[clean_header]
            else:
                for k, v in clean_columns.items():
                    if clean_header in k or k in clean_header:
                        real_col_name = v
                        break
            if real_col_name is not None:
                val = row.get(real_col_name, np.nan)
                row_data[internal_name] = val
            
        processed_rows.append(row_data)
        metadata_list.append({'pid': pid, 'orig_idx': idx})

    # Convert to DataFrame
    data_df = pd.DataFrame(processed_rows)
    
    # 2. Data Cleaning & Encoding
    print("Encoding categorical features...")
    for col in data_df.columns:
        # Try converting to numeric first
        data_df[col] = pd.to_numeric(data_df[col], errors='ignore')
        
        # If still object type (strings), encode it
        if data_df[col].dtype == 'object':
            # Use factorize to convert strings to numbers. 
            codes, uniques = pd.factorize(data_df[col])
            # Replace -1 with NaN
            codes = codes.astype(float)
            codes[codes == -1] = np.nan
            data_df[col] = codes

    # Ensure all data is numeric now
    data_df = data_df.apply(pd.to_numeric, errors='coerce')

    # Handle all-NaN columns before KNN (to prevent dropping)
    for col in data_df.columns:
        if data_df[col].isna().all():
            print(f"Warning: Column '{col}' is entirely NaN. Filling with 0.")
            data_df[col] = 0.0

    # 3. KNN Imputation
    if len(data_df) > 0:
        print("Applying KNN Imputation...")
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = imputer.fit_transform(data_df)
        data_df = pd.DataFrame(data_imputed, columns=data_df.columns)
    
        # 4. Normalization (Only for Continuous Features)
        print("Applying Min-Max Normalization to continuous features...")
        for col in CONTINUOUS_FEATURES:
            if col in data_df.columns:
                min_val = data_df[col].min()
                max_val = data_df[col].max()
                if max_val > min_val:
                    data_df[col] = (data_df[col] - min_val) / (max_val - min_val)
                else:
                    data_df[col] = 0.0 # Handle constant columns

    # Combine back
    final_results = []
    # Ensure column order matches target_columns
    if not data_df.empty:
        data_df = data_df[target_columns]
    
    for i, item in enumerate(metadata_list):
        if not data_df.empty:
            clin_data = data_df.iloc[i].tolist()
        else:
            clin_data = [0.0] * len(target_columns)
            
        final_results.append({
            'pid': item['pid'],
            'orig_idx': item['orig_idx'],
            'clin_data': clin_data
        })
        
    return final_results, target_columns

def get_bbox_from_mask(mask_data):
    # mask_data: 3D numpy array (z, y, x)
    z_indices = np.any(mask_data, axis=(1, 2))
    if not np.any(z_indices):
        return None, None, None
    
    first_slice = np.argmax(z_indices)
    last_slice = len(z_indices) - np.argmax(z_indices[::-1]) - 1
    
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
    
    for z in range(first_slice, last_slice + 1):
        if not z_indices[z]:
            continue
        rows, cols = np.where(mask_data[z] > 0)
        if len(rows) > 0:
            y_mins.append(np.min(rows))
            y_maxs.append(np.max(rows))
            x_mins.append(np.min(cols))
            x_maxs.append(np.max(cols))
            
    if not x_mins:
        return None, None, None
        
    avg_bbox = [
        int(np.mean(x_mins)),
        int(np.mean(y_mins)),
        int(np.mean(x_maxs)),
        int(np.mean(y_maxs))
    ]
    
    return first_slice, last_slice, avg_bbox

def process_patient_nifti(patient_id, folder_path, label, excel_row):
    """处理 NIfTI 格式的病人数据"""
    nii_path = os.path.join(folder_path, 'Lung.nii')
    roi_path = os.path.join(folder_path, 'ROI.nii')
    
    if not os.path.exists(nii_path):
        return False, None
        
    try:
        # Load NIfTI header only first if possible, but we need shape
        lung_img = nib.load(nii_path)
        # We need the number of slices
        shape = lung_img.shape
        # Assuming (x, y, z) or (z, y, x). Usually NIfTI is (x, y, z)
        if len(shape) == 3:
            num_slices = shape[2]
        else:
            return False, None
            
        # Load ROI
        if os.path.exists(roi_path):
            roi_img = nib.load(roi_path)
            roi_data = roi_img.get_fdata()
            # Transpose to (z, y, x) for bbox calculation
            if roi_data.ndim == 3:
                roi_data = roi_data.transpose(2, 1, 0)
        else:
            roi_data = np.zeros((num_slices, shape[1], shape[0]))
            
        # Extract Metadata
        first, last, bbox = get_bbox_from_mask(roi_data)
        
        if first is None:
            first = 0
            last = num_slices - 1
            bbox = [0, 0, shape[0], shape[1]] # Full image bbox
            
        return True, {
            'num_slice': num_slices,
            'first_appear': first,
            'last_appear': last,
            'avg_bbox': bbox
        }
    except Exception as e:
        # print(f"Error processing NIfTI {patient_id}: {e}")
        return False, None

def build_tasks(root_dir, metadata_list):
    tasks = []
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return tasks
        
    for item in metadata_list:
        pid = item['pid']
        clin_data = item['clin_data']
        orig_idx = item['orig_idx']
        
        found = False
        for label in ['0', '1']:
            label_dir = os.path.join(root_dir, label)
            if not os.path.exists(label_dir): continue
            
            for folder in os.listdir(label_dir):
                if pid.lower() == folder.lower() or pid.lower() in folder.lower():
                    # Check for exact match or substring if that's the convention
                    # Assuming P001 matches P001 or P001_xxx
                    tasks.append((pid, os.path.join(label_dir, folder), int(label), {}, clin_data, orig_idx))
                    found = True
                    break
            if found: break
    return tasks

def main():
    args = parse_args()
    train_dir = args.train_dir
    val_dir = args.val_dir
    output_csv = args.output_csv
    meta_file = args.meta_file
    
    csv_rows = []
    
    # 1. Load Metadata from Excel (Two Sheets)
    print(f"Loading clinical metadata from {meta_file}...")
    
    # Train Metadata
    print(">>> Processing Training Set Metadata (Sheet: train+test data)...")
    train_metadata, feature_names = load_and_process_metadata_excel(meta_file, sheet_name='train+test data')
    print(f"Loaded {len(train_metadata)} training patients.")
    
    # Val Metadata
    print(">>> Processing Validation Set Metadata (Sheet: external validation)...")
    val_metadata, _ = load_and_process_metadata_excel(meta_file, sheet_name='external validation')
    print(f"Loaded {len(val_metadata)} validation patients.")
    
    print(f"Active Features ({len(feature_names)}): {feature_names}")
    
    # 2. Build Tasks
    print("\nMatching NIfTI files...")
    train_tasks = build_tasks(train_dir, train_metadata)
    val_tasks = build_tasks(val_dir, val_metadata)
    
    if args.demo:
        train_tasks = train_tasks[:10]
        val_tasks = val_tasks[:10]
        print("Running in DEMO mode.")
        
    print(f"Found {len(train_tasks)} Training tasks and {len(val_tasks)} Validation tasks.")
    
    # 3. Process All Tasks
    all_tasks = []
    # Add phase info to tasks so we can recover it later? 
    # Actually, we can process them separately or together.
    # Let's process together but keep track of indices.
    
    # Process Train
    print("\nProcessing Training Images...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_results = list(tqdm(executor.map(process_patient_wrapper, train_tasks), total=len(train_tasks), desc="Train Progress"))
        
    # Process Val
    print("\nProcessing Validation Images...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        val_results = list(tqdm(executor.map(process_patient_wrapper, val_tasks), total=len(val_tasks), desc="Val Progress"))
        
    # 4. Collect Results
    def collect_rows(results, parser_tag):
        rows = []
        for result_item in results:
            pid, success, meta, clin_data, orig_idx, label = result_item
            if success:
                row_dict = {
                    'NewPatientID': pid,
                    'OriginalIndex': orig_idx,
                    'label': label,
                    'parser': parser_tag,
                    'num_slice': meta['num_slice'],
                    'first_appear': meta['first_appear'],
                    'avg_bbox': str(meta['avg_bbox']),
                    'last_appear': meta['last_appear']
                }
                for i, feat_name in enumerate(feature_names):
                    row_dict[feat_name] = clin_data[i]
                rows.append(row_dict)
        return rows

    train_rows = collect_rows(train_results, 'train')
    val_rows = collect_rows(val_results, 'val')
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    
    # 5. Oversampling (ROS) - ONLY for Train
    print("\nApplying Random Over Sampling (ROS) to Training Set...")
    if not train_df.empty:
        train_0 = train_df[train_df['label'] == 0]
        train_1 = train_df[train_df['label'] == 1]
        
        print(f"Original Train Distribution: Class 0 = {len(train_0)}, Class 1 = {len(train_1)}")
        
        if len(train_0) > 0 and len(train_1) > 0:
            target_count = max(len(train_0), len(train_1))
            
            if len(train_1) < target_count:
                print(f"Oversampling Class 1 to {target_count}...")
                train_1_over = train_1.sample(n=target_count, replace=True, random_state=42)
                train_df_balanced = pd.concat([train_0, train_1_over])
            elif len(train_0) < target_count:
                print(f"Oversampling Class 0 to {target_count}...")
                train_0_over = train_0.sample(n=target_count, replace=True, random_state=42)
                train_df_balanced = pd.concat([train_0_over, train_1])
            else:
                train_df_balanced = pd.concat([train_0, train_1])
        else:
             print("Warning: One class is empty in training set. Skipping ROS.")
             train_df_balanced = train_df
    else:
        print("Warning: Training set is empty.")
        train_df_balanced = train_df

    # 6. Combine and Save
    df_final = pd.concat([train_df_balanced, val_df])
    
    # Sort to keep some order
    if not df_final.empty:
        df_final = df_final.sort_values(by=['parser', 'OriginalIndex']).reset_index(drop=True)
    
    # Reorder columns
    fixed_cols = ['NewPatientID']
    meta_cols_end = ['label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
    final_col_order = fixed_cols + feature_names + meta_cols_end
    
    # Ensure columns exist
    existing_cols = [c for c in final_col_order if c in df_final.columns]
    df_final = df_final[existing_cols]
    
    df_final.to_csv(output_csv, index=False)
    
    # 打印详细统计信息
    print("\n" + "="*50)
    print("Metadata Generation Complete!")
    print(f"Output saved to: {output_csv}")
    print(f"Total rows in CSV: {len(df_final)}")
    print("\nDataset Split Statistics:")
    print(df_final['parser'].value_counts())
    print("\nLabel Distribution (Final):")
    print(df_final.groupby(['parser', 'label']).size())
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
