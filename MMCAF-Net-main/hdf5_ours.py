import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import pydicom
import h5py
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import nibabel as nib
import re

# 自动识别路径: 基于当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # e:\rerun2
DEFAULT_DCM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Lung')
DEFAULT_OUTPUT_H5 = os.path.join(PROJECT_ROOT, 'data', 'data.hdf5')

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DICOM to HDF5 with multi-threading (Recursive Search)')
    parser.add_argument('--dcm_root', type=str, default=DEFAULT_DCM_ROOT, help='Root directory to search for NIfTI/DICOM files')
    parser.add_argument('--output_h5', type=str, default=DEFAULT_OUTPUT_H5, help='Output path for the HDF5 file')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use')
    return parser.parse_args()

def read_nifti_data(folder_path):
    """读取 NIfTI 数据并返回 (patient_id, data)"""
    nii_path = os.path.join(folder_path, 'Lung.nii')
    if not os.path.exists(nii_path):
        return None, None

    # Extract ID from folder name (assumed PXXXX format or just folder name)
    folder_name = os.path.basename(folder_path)
    # Try to match P number first
    id_match = re.search(r'(P\d+)', folder_name)
    if id_match:
        patient_id = id_match.group(1)
    else:
        # If no P number, use folder name as ID
        patient_id = folder_name

    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        # NIfTI usually (x, y, z), we want (z, x, y) or (z, 512, 512)
        if data.ndim == 3:
            data = data.transpose(2, 1, 0)
        return patient_id, data
    except Exception as e:
        print(f"Error processing {nii_path}: {e}")
        return None, None

def read_dicom_data(args_tuple):
    """读取 DICOM 数据并返回 (patient_id, data)"""
    dirpath, filenames = args_tuple
    data_list = []
    
    dcm_files = [f for f in filenames if f.lower().endswith('.dcm') or f.lower().endswith('.dicom')]
    if not dcm_files:
        return None, None

    dcm_files.sort()

    for filename in dcm_files:
        file_path = os.path.join(dirpath, filename)
        try:
            ds = pydicom.dcmread(file_path)
            pixel_data = ds.pixel_array
            data_list.append(pixel_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not data_list:
        return None, None

    data_list = np.array(data_list)
    
    if data_list.shape[1] != 512:
        # Basic check, might skip if size varies
        # return None, None
        pass

    # Extract Patient ID
    path_parts = dirpath.split(os.sep)
    patient_id = None
    for part in path_parts:
        if 'Lung_Dx-' in part:
            patient_id = part.split('-')[-1] # A0001
            break
    
    if not patient_id:
        # Fallback to folder name if not structured
        patient_id = os.path.basename(dirpath)

    return patient_id, data_list

def main():
    args = parse_args()
    
    dcm_root = args.dcm_root
    output_h5 = args.output_h5
    num_threads = args.threads
    
    print(f"Scanning directory: {dcm_root}")
    print(f"Output HDF5: {output_h5}")
    print(f"Threads: {num_threads}")

    # 收集所有需要处理的目录
    tasks = []
    mode = 'nifti' # Default preference
    
    # Recursive search for Lung.nii
    print("Searching for 'Lung.nii' files recursively...")
    nifti_found = False
    
    for root, dirs, files in os.walk(dcm_root):
        if 'Lung.nii' in files:
            tasks.append(('nifti', root))
            nifti_found = True
            
    if not nifti_found:
        print("No 'Lung.nii' files found. Switching to DICOM search...")
        mode = 'dicom'
        for dirpath, dirnames, filenames in os.walk(dcm_root):
            if 'ALPHA' in dirpath: # Skip some known artifacts if any
                continue
            if any(f.lower().endswith('.dcm') for f in filenames):
                tasks.append(('dicom', (dirpath, filenames)))
    
    print(f"Found {len(tasks)} tasks (Mode: {mode}).")

    if os.path.exists(output_h5):
        print(f"Checking integrity of existing file: {output_h5}")
        should_delete = False
        try:
            with h5py.File(output_h5, 'r') as f:
                _ = list(f.keys())
        except Exception as e:
            print(f"Warning: Existing HDF5 file is corrupted ({e}).")
            should_delete = True
            
        if should_delete:
            print("Deleting corrupted file and starting fresh...")
            try:
                os.remove(output_h5)
            except Exception as del_e:
                print(f"Error deleting file: {del_e}")
                print("Please manually delete the file and try again.")
                return

    try:
        with h5py.File(output_h5, 'a') as h5f:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_task = {}

                for task_type, data in tasks:
                    if task_type == 'nifti':
                        future = executor.submit(read_nifti_data, data)
                    else:
                        future = executor.submit(read_dicom_data, data)
                    future_to_task[future] = data

                for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing"):
                    try:
                        patient_id, data = future.result()
                        if patient_id and data is not None:
                            if patient_id in h5f:
                                existing_shape = h5f[patient_id].shape
                                if data.shape[0] > existing_shape[0]:
                                    del h5f[patient_id]
                                    h5f.create_dataset(patient_id, data=data, compression='gzip')
                                # Else keep existing larger one
                            else:
                                h5f.create_dataset(patient_id, data=data, compression='gzip')
                    except Exception as e:
                        print(f"Task failed: {e}")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to open or write to HDF5 file: {e}")
        if "bad local heap signature" in str(e) or "addr overflow" in str(e):
            print("The file is severely corrupted. Please delete it manually if the script fails to do so.")

    print("Done!")

if __name__ == '__main__':
    main()
