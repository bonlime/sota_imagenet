import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_tools as pt


class AngularPenaltySMLoss(nn.Module):
    """PyTorch implementation of
        1. Additive Angular Margin Loss / ArcFace
        2. Large Margin Cosine Loss / CosFase
        3. SphereFace

    Args:
        in_features: Size of model discriptor
        out_features: Number of classes
        loss_type: One of {'arcface', 'sphereface', 'cosface'}
        s: Input features norm
        m1: Margin value for ArcFace
        m2: Margin value for CosFase

    Reference:
        1. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
            https://arxiv.org/pdf/1801.07698.pdf
        2. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
            https://arxiv.org/pdf/1801.09414.pdf
        3. SphereFace:
            https://arxiv.org/abs/1704.08063

    Code:
        github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
        github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/master/src/modeling/metric_learning.py

    """

    _types = ["arcface", "sphereface", "cosface"]
    # 'name': (s, m)
    _default_values = {
        "arcface": (64.0, 0.5),
        "sphereface": (64.0, 1.35),
        "cosface": (30.0, 0.4),
    }

    def __init__(self, in_features=512, out_features=3088, loss_type="arcface", s=None, m=None, criterion=None):
        super().__init__()
        assert loss_type in self._types, f"Loss type must be in ['arcface', 'sphereface', 'cosface'], got {loss_type}"

        self.s, self.m = self._default_values[loss_type]
        # Overwright default values
        self.s = self.s if not s else s
        self.m = self.m if not m else m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.weight = torch.nn.Linear(in_features, out_features, bias=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        self.loss_type = loss_type

        # Constant for numerical stability
        self.eps = 1e-7

    def forward(self, features, y_true):
        """
        Args:
            features: raw logits from the model
            y_true: Class labels, not one-hot encoded
        """
        # Normalize weight
        wf = F.linear(F.normalize(features), F.normalize(self.weight))
        # Black magic of matrix calculus
        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[y_true]) - self.m)
        elif self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[y_true]), -1.0 + self.eps, 1 - self.eps))
                + self.m
            )
        elif self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[y_true]), -1.0 + self.eps, 1 - self.eps))
            )
        else:
            raise ValueError("Unknown loss type")

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(y_true)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class AdditiveAngularMarginLoss(nn.Module):
    r"""PyTorch implementation of
        Additive Angular Margin Loss / ArcFace

    Args:
        in_features: Size of model discriptor
        out_features: Number of classes
        s: Input features norm
        m: Margin value
        criterion: One of {'cross_entropy', 'focal', 'reduced_focal'}

    Reference:
        2. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
            https://arxiv.org/pdf/1801.09414.pdf

    Code:
        github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
        github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/master/src/modeling/metric_learning.py
    """

    def __init__(self, embedding_size, num_classes, final_criterion=nn.CrossEntropyLoss(), s=10.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_classes, embedding_size)))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.final_criterion = final_criterion

    def forward(self, features, y_true):
        """
        Args:
            features: raw logits from the model
            y_true: Class labels, not one-hot encoded
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi.to(cosine), cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y_true[..., None].long(), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = self.final_criterion(output, y_true)
        return loss


class LargeMarginCosineLoss(torch.nn.Module):
    r"""PyTorch implementation of
        2. Large Margin Cosine Loss / CosFase

    Args:
        in_features: Size of model discriptor
        out_features: Number of classes
        s: Input features norm
        m: Margin value for CosFase
        criterion: One of {'cross_entropy', 'focal', 'reduced_focal'}

    Reference:
        1. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
            https://arxiv.org/pdf/1801.07698.pdf

    Code:
        github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
        github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/master/src/modeling/metric_learning.py
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40, criterion="cross_entropy"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.criterion = LOSS_FROM_NAME[criterion]

    def forward(self, features, y_true):
        """
        Args:
            features: L2 normalized logits from the model
            y_true: Class labels, not one-hot encoded
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = torch.nn.functional.linear(features, torch.nn.functional.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(features)
        one_hot.scatter_(1, y_true.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        loss = self.criterion(output, y_true)
        return loss


class ASoftMax(nn.Module):
    """
    Args:
        embedding_size (int):
        num_classes (int):
        final_criterion (nn.Module):
    Input:
        features: raw logits from the model
        y_true: Class labels, not one-hot encoded
    """

    def __init__(self, embedding_size, num_classes, final_criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.zeros(num_classes, embedding_size)))
        nn.init.xavier_uniform_(self.weight)
        self.final_criterion = final_criterion

    def forward(self, features, y_true):
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        loss = self.final_criterion(logits, y_true)
        return loss


LOSS_FROM_NAME = {
    "arcface": AdditiveAngularMarginLoss,
    # "sphereface": None,
    # "cosface": LargeMarginCosineLoss,
    # "arcface_": functools.partial(AngularPenaltySMLoss, loss_type="arcface"),
    # "cosface_": functools.partial(AngularPenaltySMLoss, loss_type="cosface"),
    # "focal": pt.losses.FocalLoss(mode="multiclass"),
    # "reduced_focal": pt.losses.FocalLoss(mode="multiclass", combine_thr=0.5, gamma=2.0),
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    # "normalized_ce": NormalizedCELoss,
    "A-softmax": ASoftMax,
}
