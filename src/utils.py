from loguru import logger
import torch.nn as nn


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
