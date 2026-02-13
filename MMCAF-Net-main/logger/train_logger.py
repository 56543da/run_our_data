import util

from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, args, dataset_len, pixel_dict):
        super(TrainLogger, self).__init__(args, dataset_len, pixel_dict)

        assert args.is_training
        assert args.iters_per_print % args.batch_size == 0, "iters_per_print must be divisible by batch_size"
        assert args.iters_per_visual % args.batch_size == 0, "iters_per_visual must be divisible by batch_size"

        self.iters_per_print = args.iters_per_print
        self.iters_per_visual = args.iters_per_visual
        self.experiment_name = args.name
        self.max_eval = args.max_eval
        self.num_epochs = args.num_epochs
        self.loss_meters = self._init_loss_meters()
        self.loss_meter = util.AverageMeter()
        
        # 新增：用于统计整个 epoch 的平均 loss
        self.epoch_loss_meter = util.AverageMeter()

    def _init_loss_meters(self):
        loss_meters = {}

        if self.do_classify:
            loss_meters['cls_loss'] = util.AverageMeter()

        return loss_meters

    def _reset_loss_meters(self):
        for v in self.loss_meters.values():
            v.reset()

    def _update_loss_meters(self, n, cls_loss=None, seg_loss=None):
        if cls_loss is not None:
            self.loss_meters['cls_loss'].update(cls_loss, n)
        if seg_loss is not None:
            self.loss_meters['seg_loss'].update(seg_loss, n)

    def _get_avg_losses(self, as_string=False):
        if as_string:
            s = ', '.join('{}: {:.3g}'.format(k, v.avg) for k, v in self.loss_meters.items())
            return s
        else:
            loss_dict = {'batch_{}'.format(k): v.avg for k, v in self.loss_meters.items()}
            return loss_dict

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter(self, inputs, cls_logits, targets, cls_loss, optimizer):
        """Log results from a training iteration."""
        cls_loss = None if cls_loss is None else cls_loss.item()
        self._update_loss_meters(inputs.size(0), cls_loss)
        
        # 更新 epoch loss 统计
        if cls_loss is not None:
            self.epoch_loss_meter.update(cls_loss, inputs.size(0))

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, {}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self._get_avg_losses(as_string=True))

            # Write all errors as scalars to the graph
            scalar_dict = self._get_avg_losses()
            scalar_dict.update({'train/lr{}'.format(i): pg['lr'] for i, pg in enumerate(optimizer.param_groups)})
            self._log_scalars(scalar_dict, print_to_stdout=False)
            self._reset_loss_meters()

            self.write(message)

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))
        # 重置 epoch loss 统计
        self.epoch_loss_meter.reset()

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}]'.format(self.epoch, time() - self.epoch_start_time))
        
        # --- Loss Comparison Plot ---
        # 获取 Train Epoch Loss
        train_loss = self.epoch_loss_meter.avg
        # 获取 Val Epoch Loss (从 metrics 中提取)
        val_loss = metrics.get('val_loss', None)
        
        if val_loss is not None:
            # 使用 add_scalars 将两条曲线画在同一张图上
            # 这里的 tag 是 'Loss/Compare'，x 轴使用 self.epoch
            self.summary_writer.add_scalars('Loss/Compare', {
                'train': train_loss,
                'val': val_loss
            }, self.epoch)
            print(f"Logged Loss/Compare to TensorBoard: train={train_loss:.4f}, val={val_loss:.4f}")
        # ----------------------------

        if not hasattr(self, 'last_ablation_auroc'):
            self.last_ablation_auroc = {
                'train': {'ImgOnly': None, 'TabOnly': None, 'Multimodal': None},
                'val': {'ImgOnly': None, 'TabOnly': None, 'Multimodal': None},
                'external': {'ImgOnly': None, 'TabOnly': None, 'Multimodal': None},
            }

        train_imgonly_auc = metrics.get('train_Ablation_ImgOnly/AUROC', None)
        train_tabonly_auc = metrics.get('train_Ablation_TabOnly/AUROC', None)
        train_multimodal_auc = metrics.get('train_Ablation_Multimodal/AUROC', None)

        val_imgonly_auc = metrics.get('val_Ablation_ImgOnly/AUROC', None)
        val_tabonly_auc = metrics.get('val_Ablation_TabOnly/AUROC', None)
        val_multimodal_auc = metrics.get('val_Ablation_Multimodal/AUROC', None)

        ext_tabonly_auc = metrics.get('External_external_test_AUROC', None)

        if train_imgonly_auc is not None:
            self.last_ablation_auroc['train']['ImgOnly'] = float(train_imgonly_auc)
        if train_tabonly_auc is not None:
            self.last_ablation_auroc['train']['TabOnly'] = float(train_tabonly_auc)
        if train_multimodal_auc is not None:
            self.last_ablation_auroc['train']['Multimodal'] = float(train_multimodal_auc)

        if val_imgonly_auc is not None:
            self.last_ablation_auroc['val']['ImgOnly'] = float(val_imgonly_auc)
        if val_tabonly_auc is not None:
            self.last_ablation_auroc['val']['TabOnly'] = float(val_tabonly_auc)
        if val_multimodal_auc is not None:
            self.last_ablation_auroc['val']['Multimodal'] = float(val_multimodal_auc)

        if ext_tabonly_auc is not None:
            self.last_ablation_auroc['external']['TabOnly'] = float(ext_tabonly_auc)

        train_show = self.last_ablation_auroc['train']
        val_show = self.last_ablation_auroc['val']
        ext_show = self.last_ablation_auroc['external']

        if any(v is not None for v in list(train_show.values()) + list(val_show.values()) + list(ext_show.values())):
            def _fmt(v):
                return 'n/a' if v is None else f'{float(v):.4f}'
            text = (
                "| Split | ImgOnly AUROC | TabOnly AUROC | Multimodal AUROC |\n"
                "|---|---:|---:|---:|\n"
                f"| Train | {_fmt(train_show.get('ImgOnly'))} | {_fmt(train_show.get('TabOnly'))} | {_fmt(train_show.get('Multimodal'))} |\n"
                f"| Val (Internal) | {_fmt(val_show.get('ImgOnly'))} | {_fmt(val_show.get('TabOnly'))} | {_fmt(val_show.get('Multimodal'))} |\n"
                f"| External | {_fmt(ext_show.get('ImgOnly'))} | {_fmt(ext_show.get('TabOnly'))} | {_fmt(ext_show.get('Multimodal'))} |\n"
            )
            self.summary_writer.add_text('AUROC/TabOnly_Compare', text, self.epoch)

            scalars = {}
            if train_show.get('ImgOnly') is not None:
                scalars['Train_ImgOnly'] = train_show.get('ImgOnly')
            if train_show.get('TabOnly') is not None:
                scalars['Train_TabOnly'] = train_show.get('TabOnly')
            if train_show.get('Multimodal') is not None:
                scalars['Train_Multimodal'] = train_show.get('Multimodal')

            if val_show.get('ImgOnly') is not None:
                scalars['Val_ImgOnly'] = val_show.get('ImgOnly')
            if val_show.get('TabOnly') is not None:
                scalars['Val_TabOnly'] = val_show.get('TabOnly')
            if val_show.get('Multimodal') is not None:
                scalars['Val_Multimodal'] = val_show.get('Multimodal')

            if ext_show.get('TabOnly') is not None:
                scalars['External_TabOnly'] = ext_show.get('TabOnly')
            if len(scalars) > 0:
                self.summary_writer.add_scalars('AUROC/TabOnly_Compare', scalars, self.epoch)

        def _keep_scalar(k):
            if '_Ablation_' in k:
                return False
            if (k.endswith('NPV') or k.endswith('PPV') or ('/NPV' in k) or ('/PPV' in k)):
                return False
            return True

        metrics_to_log = {k: v for k, v in metrics.items() if _keep_scalar(k)}
        self._log_scalars(metrics_to_log)
        self._plot_curves(curves)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
