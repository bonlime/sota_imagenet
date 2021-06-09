from loguru import logger
import torch
import torch.nn as nn
import pytorch_tools as pt


class HardNegativeWrapper(nn.Module):
    """Loss wrapper to perform the hard negative mining
    Idea is to train only on misclassified examples
    It should be much better approximation than mean for all examples

    Args:
        loss (nn.Module): loss with `none` reduction
    """

    def __init__(self, loss, hard_pct=0.02):
        super().__init__()
        logger.info(f"Using HardNegativeWrapper. hard_pct: {hard_pct}")
        self.loss = loss
        self.hard_pct = hard_pct

    def forward(self, y_pred, y_true):
        raw_loss = self.loss(y_pred, y_true)
        # take `hard_pct` from each sample so that very misclassified samples don't overweight
        hard_loss = raw_loss.topk(int(self.hard_pct * raw_loss.size(1)), sorted=False)[0].mean()
        return hard_loss.mean()


# This version (with sigmoid) doesn't work
# class FixMatchLoss(nn.Module):
#     def __init__(self, hard_weight=0.01, hard_pct=0.01):
#         super().__init__()
#         self.criterion = pt.losses.BinaryKLDivLoss(reduction="none").cuda()
#         self.hard_weight = hard_weight
#         self.hard_pct = hard_pct

#     def forward(self, y_pred, y_true):
#         y_pred = y_pred.float()
#         half_bs = y_pred.size(0) // 2
#         if y_true.dim() == 1:
#             y_true_one_hot = torch.zeros_like(y_pred)
#             y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
#         # want to do everything in full precision to avoid nan
#         with torch.cuda.amp.autocast(enabled=False):
#             raw_soft_loss = self.criterion(y_pred[:half_bs], y_pred[half_bs:].detach().sigmoid())
#             raw_hard_loss = self.criterion(y_pred[:half_bs], y_true_one_hot[half_bs:])
#         topk_n = int(self.hard_pct * y_pred.size(1))
#         # take TOPK to avoid pushing close to 0 predictions even further
#         soft_loss = raw_soft_loss.topk(topk_n, sorted=False)[0].mean()
#         hard_loss = raw_hard_loss.topk(topk_n, sorted=False)[0].mean()
#         # print(f"Hard: {hard_loss.item()}. Soft: {soft_loss.item()}")
#         return soft_loss + self.hard_weight * hard_loss


class FixMatchLoss(nn.Module):
    def __init__(self, hard_weight=0.01, hard_pct=0.01):
        super().__init__()
        self.criterion = pt.losses.BinaryKLDivLoss(reduction="none").cuda()
        self.hard_weight = hard_weight
        self.hard_pct = hard_pct

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        half_bs = y_pred.size(0) // 2
        if y_true.dim() == 1:
            y_true_one_hot = torch.zeros_like(y_pred)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        # want to do everything in full precision to avoid nan
        with torch.cuda.amp.autocast(enabled=False):
            raw_soft_loss = self.criterion(y_pred[:half_bs], y_pred[half_bs:].detach().sigmoid())
            raw_hard_loss = self.criterion(y_pred[:half_bs], y_true_one_hot[half_bs:])
        topk_n = int(self.hard_pct * y_pred.size(1))
        # take TOPK to avoid pushing close to 0 predictions even further
        soft_loss = raw_soft_loss.topk(topk_n, sorted=False)[0].mean()
        hard_loss = raw_hard_loss.topk(topk_n, sorted=False)[0].mean()
        # print(f"Hard: {hard_loss.item()}. Soft: {soft_loss.item()}")
        return soft_loss + self.hard_weight * hard_loss


# class FixMatchLossSoftMax(nn.Module):
#     """Idea is close to the loss above but use SoftMax instead"""

#     def __init__(self, hard_weight=0.01, smoothing=0.1):
#         super().__init__()
#         self.criterion = pt.losses.CrossEntropyLoss(smoothing=smoothing).cuda()
#         self.hard_weight = hard_weight

#     def forward(self, y_pred, y_true):
#         y_pred = y_pred.float()
#         half_bs = y_pred.size(0) // 2
#         if y_true.dim() == 1:
#             y_true_one_hot = torch.zeros_like(y_pred)
#             y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
#         # want to do everything in full precision to avoid nan
#         with torch.cuda.amp.autocast(enabled=False):
#             raw_soft_loss = self.criterion(y_pred[:half_bs], y_pred[half_bs:].detach().sigmoid())
#             raw_hard_loss = self.criterion(y_pred[:half_bs], y_true_one_hot[half_bs:])
#         topk_n = int(self.hard_pct * y_pred.size(1))
#         # take TOPK to avoid pushing close to 0 predictions even further
#         soft_loss = raw_soft_loss.topk(topk_n, sorted=False)[0].mean()
#         hard_loss = raw_hard_loss.topk(topk_n, sorted=False)[0].mean()
#         # print(f"Hard: {hard_loss.item()}. Soft: {soft_loss.item()}")
#         return soft_loss + self.hard_weight * hard_loss
