import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import util

from datetime import datetime
from tensorboardX import SummaryWriter

plt.switch_backend('agg')


class BaseLogger(object):
    def __init__(self, args, dataset_len, pixel_dict):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.dataset_len = dataset_len
        self.device = args.device
        self.img_format = args.img_format
        self.save_dir = args.save_dir if args.is_training else args.results_dir
        self.do_classify = args.do_classify
        self.num_visuals = args.num_visuals
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))
        
        # 修复断点续训时 TensorBoard 目录问题
        # 如果 args.ckpt_path 存在（即正在 resume），尝试复用其父目录下的 logs 文件夹（如果有的话）
        # 或者直接沿用实验名称，不加时间戳，以保证同一个实验名对应同一个 TensorBoard 曲线
        
        # 方案：直接使用固定的 logs/<experiment_name> 目录
        # 这样无论何时重启，只要 experiment_name 不变，曲线就会接上
        # log_dir = os.path.join('logs', args.name)
        
        # User requested timestamp
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H-%M-%S'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.epoch = args.start_epoch
        # Current iteration in epoch (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        # 修正：断点续训时，args.start_epoch 是加载的 epoch + 1。
        # 因此初始 global_step 应该是 (start_epoch - 1) * dataset_len
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.batch_size)
        
        # 强制修正：如果 start_epoch > 1，确保 summary_writer 不会覆盖之前的 step
        # (TensorBoard 会自动处理追加，但前提是 global_step 必须是递增的)
        self.iter_start_time = None
        self.epoch_start_time = None
        self.pixel_dict = pixel_dict

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step)

    def _plot_curves(self, curves_dict):
        """Plot all curves in a dict as RGB images to TensorBoard."""
        for name, curve in curves_dict.items():
            if name.endswith('_Ablation_Table') and isinstance(curve, str):
                self.summary_writer.add_text(name.replace('_', '/'), curve, global_step=self.global_step)
                print(f"Uploaded {name} table to TensorBoard at step {self.global_step}")
                continue
            # 特殊处理：如果已经是处理好的图像（如 SHAP_Summary），直接上传
            if (name.endswith('_SHAP_Summary') or name.endswith('_GradCAM_Overlay') or name.endswith('_Input_Samples')) and isinstance(curve, np.ndarray) and curve.ndim == 3:
                # 转换 (H, W, C) 为 (C, H, W)
                img_to_upload = np.ascontiguousarray(curve.transpose(2, 0, 1))
                self.summary_writer.add_image(name.replace('_', '/'), img_to_upload, global_step=self.global_step)
                print(f"Directly uploaded {name} image to TensorBoard.")
                continue

            fig = plt.figure()
            ax = plt.gca()

            plot_type = None
            for suffix in ['Ablation_Table', 'SHAP_Summary', 'GradCAM_Overlay', 'AttnMap', 'GradCAM', 'Confusion Matrix', 'ROC', 'PRC']:
                if name.endswith('_' + suffix):
                    plot_type = suffix
                    break
            if plot_type is None:
                plot_type = name.split('_')[-1]
            
            # 如果是 SHAP 图像但不是 numpy array (这不应该发生，因为上面已经拦截了)
            # 或者如果拦截逻辑漏掉了，这里做个兜底
            if plot_type == 'SHAP_Summary' and isinstance(curve, np.ndarray) and curve.ndim == 3:
                 # 转换 (H, W, C) 为 (C, H, W)
                img_to_upload = curve.transpose(2, 0, 1)
                self.summary_writer.add_image(name.replace('_', '/'), img_to_upload, global_step=self.global_step)
                print(f"Directly uploaded {name} image to TensorBoard (fallback).")
                plt.close(fig)
                continue

            ax.set_title(plot_type)
            if plot_type == 'PRC':
                if not (isinstance(curve, tuple) and len(curve) >= 2 and len(curve[0]) > 0 and len(curve[1]) > 0):
                    plt.close(fig)
                    continue
                precision, recall, _ = curve
                ax.step(recall, precision, color='b', alpha=0.2, where='post')
                ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
            elif plot_type == 'ROC':
                if not (isinstance(curve, tuple) and len(curve) >= 2 and len(curve[0]) > 0 and len(curve[1]) > 0):
                    plt.close(fig)
                    continue
                false_positive_rate, true_positive_rate, thresholds = curve
                # 计算 AUC
                from sklearn.metrics import auc
                roc_auc = auc(false_positive_rate, true_positive_rate)
                
                ax.plot(false_positive_rate, true_positive_rate, color='b', label=f'AUC = {roc_auc:.4f}')
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.legend(loc="lower right") # 显示 AUC 图例
            elif plot_type == 'Confusion Matrix':
                cm = curve
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title('Confusion Matrix')
                plt.colorbar(im, ax=ax)
                classes = ['Negative', 'Positive']
                
                # 显式设置刻度位置为整数索引 [0, 1]
                tick_marks = np.arange(len(classes))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)

                # 确保文字绘制在网格中心
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        # j 是 x 坐标 (列索引), i 是 y 坐标 (行索引)
                        ax.text(j, i, format(cm[i, j], 'd'),
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")

                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')
                
                # 强制设置坐标轴范围，防止裁剪或偏移
                ax.set_xlim(-0.5, cm.shape[1] - 0.5)
                ax.set_ylim(cm.shape[0] - 0.5, -0.5)
                ax.set_aspect('equal')
            elif plot_type == 'GradCAM':
                # curve is gcam numpy array
                # 简单可视化：绘制热力图
                gcam = curve
                im = ax.imshow(gcam, cmap='jet')
                ax.set_title('Grad-CAM Heatmap')
                plt.colorbar(im, ax=ax)
                ax.axis('off')
            elif plot_type == 'GradCAM_Overlay':
                ax.imshow(curve)
                ax.axis('off')
                ax.set_title('Grad-CAM Overlay')
            elif plot_type == 'AttnMap':
                # curve is attn_map numpy array [N, M]
                im = ax.imshow(curve, cmap='viridis')
                ax.set_title('Cross-Attention Weights')
                plt.colorbar(im, ax=ax)
                ax.set_xlabel('Key/Value')
                ax.set_ylabel('Query')
            elif plot_type == 'SHAP_Summary':
                # curve is shap_img numpy array [H, W, 3]
                ax.imshow(curve)
                ax.axis('off')
                ax.set_title('SHAP Feature Importance (Table)')
            else:
                ax.plot(curve[0], curve[1], color='b')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])

            fig.canvas.draw()

            # 使用更稳健的方式获取图像 buffer
            w, h = fig.canvas.get_width_height()
            curve_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            curve_img = curve_img.reshape((h, w, 3)).transpose(2, 0, 1) # 转换为 (C, H, W)
            curve_img = np.ascontiguousarray(curve_img)

            self.summary_writer.add_image(name.replace('_', '/'), curve_img, global_step=self.global_step)
            print(f"Uploaded {name} plot to TensorBoard at step {self.global_step}")

            plt.close(fig) # 必须关闭，否则会占用大量内存

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            try:
                from tqdm import tqdm
                tqdm.write(message)
            except Exception:
                print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError
