# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/RN50v1.5/image_classification/mixup.py
import torch
import torch.nn as nn
import numpy as np


def mixup(alpha, num_classes, data, target):
    with torch.no_grad():
        if len(target.shape) == 1: # if not one hot
            target_one_hot = torch.zeros(target.size(0), num_classes, 
                                         dtype=torch.float, device=data.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        else:
            target_one_hot = target
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(bs).cuda()

        md = c * data + (1-c) * data[perm, :]
        mt = c * target_one_hot + (1-c) * target_one_hot[perm, :]
        return md, mt


class MixUpWrapper:
    def __init__(self, alpha, num_classes, dataloader):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes

    def mixup_loader(self, loader):
        for input, target in loader:
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)