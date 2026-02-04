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
DEFAULT_DCM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Lung')
DEFAULT_METADATA_FILE = os.path.join(PROJECT_ROOT, 'data', '原始STAS_data.xlsx')
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'G_first_last_nor.csv')

# 设置随机种子以保证可复现性
random.seed(42)

# Feature Configuration
# Continuous features to be normalized
CONTINUOUS_FEATURES = [
    'Age', 'Weight'
]

# Categorical/Ordinal features (no normalization, just encoding)
CATEGORICAL_FEATURES = [
    'Sex', 'Smoking'
]

# Mapping from Excel headers to Internal names (Fuzzy matching will be used)
COLUMN_MAPPING = {
    '性别': 'Sex',
    '年龄': 'Age',
    'BMI': 'Weight',
    '吸烟史': 'Smoking',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data metadata (NIfTI + Excel mode)')
    parser.add_argument('--dcm_root', type=str, default=DEFAULT_DCM_ROOT, help='Root directory of NIfTI files')
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

def load_and_process_metadata_excel(meta_path):
    """读取并处理临床表格数据 (Excel版)"""
    if not os.path.exists(meta_path):
        print(f"Warning: Metadata file not found at {meta_path}")
        return []

    try:
        if meta_path.endswith('.csv'):
             df = pd.read_csv(meta_path)
        else:
             df = pd.read_excel(meta_path)
        # Normalize columns: strip whitespace and remove internal spaces for matching
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return []
    
    # Identify Name column for ID extraction
    name_col = None
    possible_name_cols = ['Name', '姓名', 'PatientName', 'PatientsName']
    for col in df.columns:
        if any(x.lower() == str(col).lower() for x in possible_name_cols):
             name_col = col
        elif any(x in str(col) for x in possible_name_cols):
             if not name_col: name_col = col
             
    if not name_col:
        print("Warning: Could not find Name column in Excel.")
        return []

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
            if clean_header in clean_columns:
                real_col_name = clean_columns[clean_header]
                val = row.get(real_col_name, np.nan)
                row_data[internal_name] = val
            
        processed_rows.append(row_data)
        metadata_list.append({'pid': pid, 'orig_idx': idx})

    # Convert to DataFrame
    data_df = pd.DataFrame(processed_rows)
    
    # Check if we actually got data for TumorSize
    if 'TumorSize' in data_df.columns:
        valid_count = data_df['TumorSize'].notna().sum()
        print(f"Successfully extracted 'TumorSize' with {valid_count} valid entries.")
    
    # 2. Data Cleaning & Encoding
    print("Encoding categorical features...")
    for col in data_df.columns:
        # Try converting to numeric first
        data_df[col] = pd.to_numeric(data_df[col], errors='ignore')
        
        # If still object type (strings), encode it
        if data_df[col].dtype == 'object':
            # Use factorize to convert strings to numbers. 
            # Note: factorize maps NaNs to -1. We need to keep NaNs as NaNs for Imputer.
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
        # Use values to avoid column mismatch if version differs
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
    data_df = data_df[target_columns]
    
    for i, item in enumerate(metadata_list):
        clin_data = data_df.iloc[i].tolist()
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
        # We need to be consistent. Let's assume z is last dim if 3D
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

def main():
    args = parse_args()
    dcm_root = args.dcm_root
    output_csv = args.output_csv
    meta_file = args.meta_file
    csv_rows = []
    
    # Mode Selection: NIfTI Processing vs CSV Update
    use_existing_csv = False
    if not os.path.exists(dcm_root):
        print(f"NIfTI root directory not found: {dcm_root}")
        if os.path.exists(output_csv):
            print(f"Switching to Metadata Update Mode using existing CSV: {output_csv}")
            use_existing_csv = True
        else:
            print("Error: Neither NIfTI directory nor existing CSV found. Cannot proceed.")
            return
        
    # Load clinical metadata (Excel)
    print(f"Loading clinical metadata from {meta_file}...")
    excel_metadata, feature_names = load_and_process_metadata_excel(meta_file)
    if not excel_metadata: return
    print(f"Loaded metadata for {len(excel_metadata)} patients.")
    print(f"Active Features ({len(feature_names)}): {feature_names}")
    
    if use_existing_csv:
        # Load existing CSV to recover image metadata
        existing_df = pd.read_csv(output_csv)
        image_meta_map = {}
        # Iterate to find unique metadata per patient
        for _, row in existing_df.iterrows():
            pid = str(row['NewPatientID'])
            if pid not in image_meta_map:
                image_meta_map[pid] = {
                    'num_slice': row['num_slice'],
                    'first_appear': row['first_appear'],
                    'avg_bbox': row['avg_bbox'],
                    'last_appear': row['last_appear'],
                    'label': row['label']
                }
        print(f"Recovered image metadata for {len(image_meta_map)} patients from existing CSV.")
        
        # Merge
        for item in excel_metadata:
            pid = str(item['pid'])
            clin_data = item['clin_data']
            orig_idx = item['orig_idx']
            
            if pid in image_meta_map:
                meta = image_meta_map[pid]
                rand_val = random.random()
                parser = 'train' if rand_val < 0.7 else 'val'
                
                row_dict = {
                    'NewPatientID': pid,
                    'OriginalIndex': orig_idx,
                    'label': meta['label'],
                    'parser': parser,
                    'num_slice': meta['num_slice'],
                    'first_appear': meta['first_appear'],
                    'avg_bbox': meta['avg_bbox'],
                    'last_appear': meta['last_appear']
                }
                
                for i, feat_name in enumerate(feature_names):
                    row_dict[feat_name] = clin_data[i]
                csv_rows.append(row_dict)
            else:
                # print(f"Skipping {pid} (No image data in CSV)")
                pass

    else:
        # Build tasks in Excel order (Original NIfTI Logic)
        tasks = []
        for item in excel_metadata:
            pid = item['pid']
            clin_data = item['clin_data']
            orig_idx = item['orig_idx']
            
            found = False
            for label in ['0', '1']:
                label_dir = os.path.join(dcm_root, label)
                if not os.path.exists(label_dir): continue
                
                for folder in os.listdir(label_dir):
                    if pid.lower() in folder.lower():
                        tasks.append((pid, os.path.join(label_dir, folder), int(label), {}, clin_data, orig_idx))
                        found = True
                        break
                    if '_' in pid and pid.lower() in folder.lower():
                         tasks.append((pid, os.path.join(label_dir, folder), int(label), {}, clin_data, orig_idx))
                         found = True
                         break
                if found: break
            
        if args.demo:
            tasks = tasks[:20]
            print("Running in DEMO mode (first 20 patients).")
        else:
            print("Running in FULL mode (processing all patients).")
        
        print(f"Found {len(tasks)} tasks. Starting preprocessing...")
        
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_patient_wrapper, tasks), total=len(tasks), desc="Overall Progress"))

        for task, result_item in zip(tasks, results):
            pid, success, meta, clin_data, orig_idx, label = result_item
            short_id = pid
            
            if success:
                rand_val = random.random()
                parser = 'train' if rand_val < 0.7 else 'val'
                
                row_dict = {
                    'NewPatientID': short_id,
                    'OriginalIndex': orig_idx,
                    'label': label,
                    'parser': parser,
                    'num_slice': meta['num_slice'],
                    'first_appear': meta['first_appear'],
                    'avg_bbox': str(meta['avg_bbox']),
                    'last_appear': meta['last_appear']
                }
                
                for i, feat_name in enumerate(feature_names):
                    row_dict[feat_name] = clin_data[i]
                    
                csv_rows.append(row_dict)
    
    df = pd.DataFrame(csv_rows)
    
    # --- Random Over Sampling (ROS) Logic ---
    print("\nApplying Random Over Sampling (ROS) to Training Set...")
    # Separate Train and Val
    train_df = df[df['parser'] == 'train']
    val_df = df[df['parser'] == 'val']
    
    # Analyze Train Distribution
    train_0 = train_df[train_df['label'] == 0]
    train_1 = train_df[train_df['label'] == 1]
    
    print(f"Original Train Distribution: Class 0 (Negative) = {len(train_0)}, Class 1 (Positive) = {len(train_1)}")
    
    target_count = max(len(train_0), len(train_1))
    
    if len(train_1) < target_count:
        print(f"Oversampling Class 1 (Positive) from {len(train_1)} to {target_count}...")
        train_1_over = train_1.sample(n=target_count, replace=True, random_state=42)
        train_df_balanced = pd.concat([train_0, train_1_over])
    else:
        train_df_balanced = pd.concat([train_0, train_1])
        
    if len(train_0) < target_count:
         print(f"Oversampling Class 0 (Negative) from {len(train_0)} to {target_count}...")
         train_0_over = train_0.sample(n=target_count, replace=True, random_state=42)
         train_df_balanced = pd.concat([train_0_over, train_1] if len(train_1) >= target_count else [train_0_over, train_1_over])

    # NOTE: To maintain Excel order as much as possible, we sort by OriginalIndex.
    print("Sorting final dataframe by original Excel order...")
    df_balanced = pd.concat([train_df_balanced, val_df])
    df_balanced = df_balanced.sort_values(by=['OriginalIndex', 'label']).reset_index(drop=True)
    
    # Reorder columns to ensure features are at indices 1-N (flexible)
    # Fixed metadata first
    fixed_cols = ['NewPatientID']
    # Clinical features next
    # Metadata last
    meta_cols_end = ['label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
    
    # The feature_names are already in clin_data order
    final_col_order = fixed_cols + feature_names + meta_cols_end
    
    # Check if all columns exist (they should)
    df_final = df_balanced[final_col_order]
    df_final.to_csv(output_csv, index=False)
    
    # 打印详细统计信息
    print("\n" + "="*50)
    print("Metadata Generation Complete!")
    print(f"Output saved to: {output_csv}")
    print(f"Total rows in CSV (including ROS): {len(df_final)}")
    print(f"Clinical Features ({len(feature_names)}):")
    print(feature_names)
    print("\nDataset Split Statistics:")
    print(df_final['parser'].value_counts())
    print("\nLabel Distribution:")
    print(df_final['label'].value_counts().rename({0: 'STAS-Negative', 1: 'STAS-Positive'}))
    print("="*50 + "\n")

if __name__ == '__main__':
    main()