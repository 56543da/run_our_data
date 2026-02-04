import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification.
    
    Args:
        gamma (float): Focusing parameter. Default: 2.
        alpha (float): Weighting factor for positive class (1). Default: 0.25 (reduce weight of easy negatives).
                       If set to None, no alpha weighting is applied.
                       If set to > 0.5, it increases the weight of positive class.
    """
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.take_mean = size_average

    def forward(self, logits, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        # Standard Cross Entropy part (BCEWithLogits)
        # max_val = (-logits).clamp(min=0)
        # loss = logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()
        
        # 使用 PyTorch 原生函数计算 BCE，数值更稳定
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # Pt: probability of being classified correctly
        # if target=1, pt = p; if target=0, pt = 1-p
        pred_prob = torch.sigmoid(logits)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            # Focal Loss formula: -alpha * (1-pt)^gamma * log(pt)
            # bce_loss is -log(pt)
            loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss
        else:
            loss = (1 - pt).pow(self.gamma) * bce_loss

        if self.take_mean:
            loss = loss.mean()

        return loss
