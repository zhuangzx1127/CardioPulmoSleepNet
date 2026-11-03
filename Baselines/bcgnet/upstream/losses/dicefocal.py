from torch import nn
import torch.nn.functional as F
from monai.losses import DiceFocalLoss, FocalLoss, DiceLoss, TverskyLoss


class DiceFocalLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True, gamma=2.0, lambda_dice=0.5, lambda_focal=0.5, reduction='mean')

    def forward(self, prediction, target):
        prediction = prediction['ahi_event']
        # prediction = prediction.clamp(min=-10, max=10)
        target = target['ah_event']
        if len(prediction.size()) == 3:
            prediction = prediction.squeeze()
        loss = self.loss(prediction.unsqueeze(1), target.unsqueeze(1))
        losses = {
            'head_loss': loss,
            'iou_loss': 0.0,
            'loss': loss,
        }
        return losses
 
class DiceFocalLossWrapperV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True, reduction='mean')
        self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, use_softmax =False, gamma=2.0)

    def forward(self, prediction, target):
        prediction = prediction['ahi_event']
        # prediction = prediction.clamp(min=-10, max=10)
        target = target['ah_event']
        if len(prediction.size()) == 3:
            prediction = prediction.squeeze()
        dice_loss = self.dice_loss(prediction.unsqueeze(1)[:,:,:,None], target.unsqueeze(1)[:,:,:,None])
        focal_loss = self.focal_loss(prediction.unsqueeze(1), target.unsqueeze(1))
        loss = 0.5 * dice_loss + 0.5 * focal_loss
        losses = {
            'head_loss': loss,
            'iou_loss': 0.0,
            'loss': loss,
        }
        return losses

class TverskyFocalLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky_loss = TverskyLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True, reduction='mean')
        self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, use_softmax =False, gamma=2.0)

    def forward(self, prediction, target):
        prediction = prediction['ahi_event']
        # prediction = prediction.clamp(min=-10, max=10)
        target = target['ah_event']
        if len(prediction.size()) == 3:
            prediction = prediction.squeeze()
        tversky_loss = self.tversky_loss(prediction.unsqueeze(1), target.unsqueeze(1))
        focal_loss = self.focal_loss(prediction.unsqueeze(1), target.unsqueeze(1))
        loss = 0.5 * tversky_loss + 0.5 * focal_loss
        losses = {
            'head_loss': loss,
            'iou_loss': 0.0,
            'loss': loss,
        }
        return losses

def smooth_binary_labels(labels, smoothing=0.1):
    """
    输入:
        labels: shape (batch_size,), 元素为 0 或 1
        smoothing: 平滑系数，例如 0.1
    输出:
        smoothed_labels: shape (batch_size,), 值在 (0, 1) 之间
    """
    labels = labels.float()
    return labels * (1 - smoothing) + 0.5 * smoothing

class FocalLossWrapper(nn.Module):
    def __init__(self, smoothing: float | None = None):
        super().__init__()
        self.loss = FocalLoss(include_background=True, to_onehot_y=False, use_softmax =False, gamma=2.0, alpha=0.25)
        self.smoothing = smoothing

    def forward(self, prediction, target):
        prediction = prediction['ahi_event']
        # prediction = prediction.clamp(min=-10, max=10)
        target = target['ah_event']
        if self.smoothing is not None:
            target = smooth_binary_labels(target, self.smoothing)
        if len(prediction.size()) == 3:
            prediction = prediction.squeeze()
        loss = self.loss(prediction.unsqueeze(1), target.unsqueeze(1))
        losses = {
            'head_loss': loss,
            'iou_loss': 0.0,
            'loss': loss,
        }
        return losses