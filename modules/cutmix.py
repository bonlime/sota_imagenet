import torch
import torch.nn as nn
import numpy as np


class CutMixWrapper:
    def __init__(self, alpha, num_classes, loader):
        self.tb = torch.distributions.Beta(alpha, alpha)
        self.loader = loader
        self.num_classes = num_classes

    def __iter__(self):
        return (self.cutmix(d, t) for d, t in self.loader)

    def cutmix(self, data, target):
        with torch.no_grad():
            if len(target.shape) == 1:  # if not one hot
                target_one_hot = torch.zeros(
                    target.size(0), self.num_classes, dtype=torch.float, device=data.device
                )
                target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            else:
                target_one_hot = target
            BS, C, H, W = data.size()
            perm = torch.randperm(BS).cuda()
            lam = self.tb.sample()
            lam = min([lam, 1 - lam])
            bbh1, bbw1, bbh2, bbw2 = self.rand_bbox(H, W, lam)
            # real lambda may be diffrent from sampled. adjust for it
            lam = (bbh2 - bbh1) * (bbw2 - bbw1) / (H * W)
            data[:, bbh1:bbh2, bbw1:bbw2] = data[perm, bbh1:bbh2, bbw1:bbw2]
            mixed_target = (1 - lam) * target_one_hot + lam * target_one_hot[perm, :]
        return data, mixed_target

    @staticmethod
    def rand_bbox(H, W, lam):
        """ returns bbox with area close to lam*H*W """
        cut_rat = np.sqrt(lam)
        cut_h = np.int(H * cut_rat)
        cut_w = np.int(W * cut_rat)
        # uniform
        ch = np.random.randint(H)
        cw = np.random.randint(W)
        bbh1 = np.clip(ch - cut_h // 2, 0, H)
        bbw1 = np.clip(cw - cut_w // 2, 0, W)
        bbh2 = np.clip(ch + cut_h // 2, 0, H)
        bbw2 = np.clip(cw + cut_w // 2, 0, W)
        return bbh1, bbw1, bbh2, bbw2

    def __len__(self):
        return len(self.loader)
