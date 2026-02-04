import torch
import torch.nn.functional as F

from collections import OrderedDict
from .base_cam import BaseCAM


class GradCAM(BaseCAM):
    """Class for generating grad CAMs.

    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device, is_binary, is_3d):
        super(GradCAM, self).__init__(model, device, is_binary, is_3d)
        self.fmaps = OrderedDict()
        self.grads = OrderedDict()
        self.hooks = []

    def _register_hooks(self, target_layer):
        """对目标层注册 hook，使用 tensor.register_hook 以规避 inplace 报错"""
        self._release_hooks()
        
        # 处理 DataParallel 情况
        model = self.model.module if hasattr(self.model, 'module') else self.model

        def save_fmaps_and_grads(m, _, output):
            # 获取输出 tensor
            if isinstance(output, (tuple, list)):
                out_tensor = output[0]
            else:
                out_tensor = output
            
            # 保存特征图 (必须 clone 避免 inplace 修改影响)
            self.fmaps[target_layer] = out_tensor.detach().cpu().clone()
            
            # 注册 tensor 级别的 backward hook
            def save_grad(grad):
                if grad is not None:
                    self.grads[target_layer] = grad.detach().cpu().clone()
            
            # 为 tensor 注册 hook，这比 register_backward_hook 更鲁棒
            self.hooks.append(out_tensor.register_hook(save_grad))

        found = False
        for name, module in model.named_modules():
            if name == target_layer:
                # 仅注册 forward hook，backward 通过 tensor.register_hook 实现
                self.hooks.append(module.register_forward_hook(save_fmaps_and_grads))
                found = True
                break
        
        if not found:
            # 尝试自动补全前缀（针对 DataParallel）
            for name, module in model.named_modules():
                if name.endswith('.' + target_layer) or target_layer.endswith('.' + name):
                    self.hooks.append(module.register_forward_hook(save_fmaps_and_grads))
                    found = True
                    break

        if not found:
            raise ValueError(f'Invalid layer name: {target_layer}')

    def _release_hooks(self):
        """释放所有 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.fmaps.clear()
        self.grads.clear()

    @staticmethod
    def _normalize(grads):
        return grads / (torch.norm(grads).item() + 1e-5)

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        if self.is_3d:
            weights = F.adaptive_avg_pool3d(grads, 1)
        else:
            weights = F.adaptive_avg_pool2d(grads, 1)
        return weights

    def get_cam(self, target_layer):
        # 注意：现在 hook 只针对 target_layer，所以直接从 dict 取
        if target_layer not in self.fmaps or target_layer not in self.grads:
            raise ValueError(f"No data for {target_layer}. Did you run forward/backward?")
            
        fmaps = self.fmaps[target_layer]
        grads = self.grads[target_layer]
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)

        gcam -= gcam.min()
        if gcam.max() > 0:
            gcam /= gcam.max()

        return gcam.detach().numpy()
