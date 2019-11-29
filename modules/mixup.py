# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/RN50v1.5/image_classification/mixup.py
import torch
import torch.nn as nn


class MixUpWrapper:
    def __init__(self, alpha, num_classes, loader):
        self.tb = torch.distributions.Beta(alpha, alpha)
        self.loader = loader
        self.num_classes = num_classes

    def __iter__(self):
        return (self.mixup(d, t) for d, t in self.loader)

    def mixup(self, data, target):
        with torch.no_grad():
            if len(target.shape) == 1:  # if not one hot
                target_one_hot = torch.zeros(
                    target.size(0), self.num_classes, dtype=torch.float, device=data.device
                )
                target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            else:
                target_one_hot = target
            bs = data.size(0)
            c = self.tb.sample()
            perm = torch.randperm(bs).cuda()
            md = c * data + (1 - c) * data[perm, :]
            mt = c * target_one_hot + (1 - c) * target_one_hot[perm, :]
            return md, mt

    def __len__(self):
        return len(self.loader)
