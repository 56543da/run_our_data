import cv2
import h5py
import numpy as np
import os
import pickle
import torch
import util
import random
import albumentations
from scipy.ndimage.interpolation import rotate
import sys
#w
from ct.ct_pe_constants import *
from .base_ct_dataset import BaseCTDataset

class CTPEDataset3d(BaseCTDataset):
    def __init__(self, args, phase, is_training_set = True):
        super(CTPEDataset3d, self).__init__(args.data_dir, args.img_format, is_training_set = is_training_set)
        self.phase = phase
        self.resize_shape = args.resize_shape
        self.is_test_mode = not args.is_training
        self.pe_types = args.pe_types
        #w
        self.crop_shape = args.crop_shape
        self.use_bbox_crop = getattr(args, 'use_bbox_crop', False)
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate
        self.do_jitter = self.is_training_set and args.do_jitter
        #w 默认为真
        self.pixel_dict = {
            'min_val':CONTRAST_HU_MIN,
            'max_val':CONTRAST_HU_MAX,
            'avg_val':CONTRAST_HU_MEAN,
            'w_center': W_CENTER_DEFAULT,
            'w_width': W_WIDTH_DEFAULT
        }
        #w
        with open(args.pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)

        self.h5_filename = getattr(args, 'h5_filename', None) or 'data.hdf5'

        # Filter out samples that are not in the HDF5 file
        h5_path = os.path.join(self.data_dir, self.h5_filename)
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                valid_keys = set(f.keys())
            
            valid_ctpes = []
            filtered_count = 0
            missing_ctpes = []
            for ctpe in all_ctpes:
                if str(ctpe.study_num) in valid_keys:
                    valid_ctpes.append(ctpe)
                else:
                    filtered_count += 1
                    missing_ctpes.append(ctpe)
            
            if filtered_count > 0:
                print(f"Filtered {filtered_count} samples that are missing in {self.h5_filename}. Remaining: {len(valid_ctpes)}")
                missing_report_path = os.path.join(
                    self.data_dir,
                    f"missing_in_{os.path.splitext(self.h5_filename)[0]}_{self.phase}.txt"
                )
                try:
                    with open(missing_report_path, "w", encoding="utf-8") as fh:
                        fh.write(f"h5_path={h5_path}\n")
                        fh.write(f"phase={self.phase}\n")
                        fh.write(f"missing_count={len(missing_ctpes)}\n")
                        fh.write("missing_items:\n")
                        for ctpe in missing_ctpes:
                            fh.write(
                                f"- study_num={ctpe.study_num}\t"
                                f"label={int(ctpe.is_positive)}\t"
                                f"parser={ctpe.phase}\t"
                                f"num_slice={ctpe.num_slice}\t"
                                f"first_appear={ctpe.first_appear}\t"
                                f"last_appear={ctpe.last_appear}\n"
                            )
                    print(f"Missing sample details written to: {missing_report_path}")
                except Exception as e:
                    print(f"Warning: failed to write missing sample report to {missing_report_path}: {e}")

                print("Missing study_num list (first 50):")
                for ctpe in missing_ctpes[:50]:
                    print(f"  - {ctpe.study_num} (label={int(ctpe.is_positive)}, parser={ctpe.phase}, num_slice={ctpe.num_slice})")
            all_ctpes = valid_ctpes
        else:
            print(f"Warning: HDF5 file not found at {h5_path}")

        #w
        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)] #w 根据phase筛选样本
        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.num_slices = args.num_slices
        #w
        self.window_to_series_idx = []  
        self.series_to_window_idx = []  
        window_start = 0
        for i, s in enumerate(self.ctpe_list): 
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0) 
            self.window_to_series_idx += num_windows * [i]   
            self.series_to_window_idx.append(window_start) 
            window_start += num_windows 
        print(f"{self.phase} windows: {len(self.window_to_series_idx)} (series={len(self.ctpe_list)}, num_slices={self.num_slices})")
        #sys.exit()
    #w
    def _include_ctpe(self, pe):
        if pe.phase != self.phase:
            return False
        '''if pe.is_positive and pe.type not in self.pe_types:
            return False'''
        return True
    #w
    def __len__(self):
        return len(self.window_to_series_idx)
    #w
    def __getitem__(self, idx):
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx] 
        #w 我们只选择能包含病灶切片的开始ID
        do_center_abnormality = random.random() < 0.5
        start_idx = self._get_abnormal_start_idx(ctpe, do_center = do_center_abnormality)
        #w
        if self.do_jitter:  
            start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)
            start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)
        #w
        volume = self._load_volume(ctpe, start_idx)   
        volume = self._transform(volume, ctpe.bbox)     
        #w
        target = {'is_abnormal':ctpe.is_positive,
                  'study_num':ctpe.study_num,   
                  'dset_path':str(ctpe.study_num),
                  'slice_idx':start_idx,  
                  'series_idx':ctpe_idx,
                  'bbox':ctpe.bbox}   
        #w
        return volume, target
    #w
    def get_series(self, study_num):
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None
    #w
    def get_series_label(self, series_idx):
        series_idx = int(series_idx)
        return float(self.ctpe_list[series_idx].is_positive)
    #w
    def _get_abnormal_start_idx(self, ctpe, do_center):
        abnormal_bounds = (ctpe.first_appear, ctpe.last_appear)
        if do_center:
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            #w 删掉了self.min_pe_slices的使用
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices,
                                       abnormal_bounds[1] + 1)
        return start_idx
    #w
    def _load_volume(self, ctpe, start_idx):    
        try:
            # 使用 getattr 获取文件名，防止属性丢失报错
            h5_filename = getattr(self, 'h5_filename', 'data.hdf5')
            
            with h5py.File(os.path.join(self.data_dir, h5_filename), 'r') as hdf5_fh:     
                key = str(ctpe.study_num)
                if key not in hdf5_fh:
                    print(f"Error: Key {key} not found in {h5_filename}")
                    return np.zeros((self.num_slices, self.resize_shape[0], self.resize_shape[1]))
                
                volume = hdf5_fh[key][start_idx:start_idx + self.num_slices, :, :]   
            return volume
        except Exception as e:
            print(f"Error loading volume for {ctpe.study_num}: {e}")
            return np.zeros((self.num_slices, self.resize_shape[0], self.resize_shape[1]))
    #w
    def _crop(self, volume, x1, y1, x2, y2):
        volume = volume[:, y1:y2, x1:x2]
        return volume
    #w
    def _rescale(self, volume, interpolation = cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)
    #w
    def _pad(self, volume):
        def add_padding(volume_, pad_value = AIR_HU_VAL):
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode = 'constant', constant_values = pad_value)
            return volume_
        #w
        volume_num_slices = volume.shape[0]
        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value = AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]
        return volume
    #w
    def _transform(self, inputs, bbox=None):
        inputs = self._pad(inputs)
        
        # 记录原始尺寸和缩放比例
        scale_h, scale_w = 1.0, 1.0
        if self.resize_shape is not None:
            orig_h, orig_w = inputs.shape[-2], inputs.shape[-1]
            inputs = self._rescale(inputs, interpolation = cv2.INTER_AREA)
            # 计算缩放因子 (new / old)
            new_h, new_w = inputs.shape[-2], inputs.shape[-1]
            if orig_h > 0 and orig_w > 0:
                scale_h = new_h / orig_h
                scale_w = new_w / orig_w

        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])    
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            
            # 默认逻辑
            if self.is_training_set:
                row = random.randint(0, row_margin)
                col = random.randint(0, col_margin)
            else:
                row = row_margin // 2
                col = col_margin // 2
            
            # BBox 逻辑: 仅在 use_bbox_crop 开启且 bbox 有效时覆盖默认逻辑
            if self.use_bbox_crop and bbox is not None:
                try:
                    # 解析 bbox
                    if isinstance(bbox, str):
                        bbox = eval(bbox)
                    
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        # 假设 bbox 是 [x1, y1, x2, y2]
                        b_x1, b_y1, b_x2, b_y2 = bbox
                        
                        # 缩放 bbox 到当前图像尺寸
                        b_x1 *= scale_w
                        b_x2 *= scale_w
                        b_y1 *= scale_h
                        b_y2 *= scale_h
                        
                        # 计算 bbox 中心点
                        c_x = (b_x1 + b_x2) / 2
                        c_y = (b_y1 + b_y2) / 2
                        
                        # 计算目标 crop 的左上角坐标 (row=y, col=x)
                        # 我们希望 crop 框的中心对齐 bbox 中心
                        target_row = int(c_y - self.crop_shape[-2] / 2)
                        target_col = int(c_x - self.crop_shape[-1] / 2)
                        
                        # 限制在合法范围内 (row_margin 是允许的最大起始点)
                        row = min(max(0, target_row), row_margin)
                        col = min(max(0, target_col), col_margin)
                        
                        # 在训练集可以增加一点随机扰动，增强鲁棒性
                        if self.is_training_set and self.do_jitter:
                            jitter_range = 10 # 像素
                            row += random.randint(-jitter_range, jitter_range)
                            col += random.randint(-jitter_range, jitter_range)
                            row = min(max(0, row), row_margin)
                            col = min(max(0, col), col_margin)
                except Exception as e:
                    # 静默失败，回退到默认逻辑
                    pass

            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])
        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis = -2)
        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis = -1)
        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape = False, cval = AIR_HU_VAL)
        #w Sharpening (Paper implementation)
        if self.is_training_set:
            inputs = self._sharpen(inputs)
            
        inputs = self._normalize_raw(inputs)
        inputs = np.expand_dims(inputs, axis = 0)  
        inputs = torch.from_numpy(inputs)
        return inputs

    def _sharpen(self, volume):
        """Apply 2D sharpening to each slice in the volume"""
        # Standard sharpening kernel
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        
        sharpened = np.empty_like(volume)
        for i in range(volume.shape[0]):
            # cv2.filter2D works on numpy arrays and preserves depth if ddepth=-1
            sharpened[i] = cv2.filter2D(volume[i], -1, kernel)
        return sharpened
