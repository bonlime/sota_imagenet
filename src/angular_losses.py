import math
import functools
import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_tools as pt
from pytorch_tools.losses import Loss


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

    def __init__(self, final_criterion=nn.CrossEntropyLoss(), s=10.0, m=0.2):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.final_criterion = final_criterion

    def forward(self, cosine, y_true):
        """
        Args:
            features: already sphere normalized logits
            y_true: Class labels, not one-hot encoded
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
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


class SphereLinearLayer(nn.Module):
    """
    Almost the same as default linear layer but performs normalization on unit hyper-sphere
    """

    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.zeros(num_classes, embedding_size)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x


class AdaCos(nn.Module):
    """PyTorch implementation of AdaCos. See Ref[1] for paper

    This implementation is different from the most open-source implementations in following ways:
    1) expects raw logits of size (bs x num_classes) not (bs, embedding_size)
    2) despite AdaCos being dynamic, still add an optional margin parameter
    3) calculate running average stats of B and θ, not batch-wise stats as in original paper
    4) normalize input logits, not embeddings and weights

    Args:
        margin (float): margin in radians
        momentum (float): momentum for running average of B and θ

    Input:
        y_pred (torch.Tensor): shape BS x N_classes
        y_true (torch.Tensor): one-hot encoded. shape BS x N_classes
    Reference:
        [1] Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations

    """

    def __init__(self, final_criterion, margin=0, max_s=20, momentum=0.95):
        super(AdaCos, self).__init__()
        self.final_criterion = final_criterion
        self.margin = margin
        self.momentum = momentum
        self.prev_s = max_s
        self.running_B = 1000  # default value is chosen so that initial S is ~10
        self.running_cos = 0.7 # 0.7 is ~= cos(pi / 4)
        self.eps = 1e-7
        self.idx = 0
        self.max_s = max_s

    def forward(self, cosine, y_true):
        # cos_theta = cosine.clamp(-1 + self.eps, 1 - self.eps)
        # cos_theta = torch.cos(torch.acos(cos_theta + self.margin))

        if y_true.dim() == 1: # if y_true is indexes
            y_true_one_hot = torch.zeros_like(cosine)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
            y_true = y_true.long()
        else:
            y_true_one_hot = y_true.float().clone()
            y_true = y_true_one_hot.argmax(-1).long()

        
        with torch.no_grad():
            B_batch = cosine[y_true_one_hot.eq(0)].mul(self.prev_s).exp().sum().div(cosine.size(0))
            # median makes more sense than mean
            # median on cosine is the same as median on angle but on cosine is faster
            # in case of cutmix take larger image class. 0.7 is ~= cos (pi / 4). needed for better convergence at the begining
            med_cos = cosine.gather(dim=1, index=y_true[..., None]).median() 

            self.running_B = self.running_B * self.momentum + B_batch * (1 - self.momentum)
            self.running_cos = self.running_cos * self.momentum + med_cos * (1 - self.momentum)
            # self.prev_s = B_batch.log() / (med_cos - self.margin)
            self.prev_s = self.running_B.log() / (self.running_cos.clamp_min(0.7) - self.margin)
            self.prev_s = min(self.prev_s, self.max_s) # limit maximum possible s. without this hack it blows up in first epochs 

        cosine = cosine.scatter_add(dim=1, index=y_true[..., None], src=torch.ones_like(cosine).neg() * self.margin)
        self.idx += 1
        if self.idx % 1000 == 0:
            logger.info(
                f"\nRunning B: {self.running_B:.2f}. Running theta: {self.running_cos:.2f}. Running S: {self.prev_s:.2f}"
            )

        return self.final_criterion(cosine * self.prev_s, y_true_one_hot)


class AdaCosMargin(nn.Module):
    """PyTorch implementation of modification of AdaCos which also supports additive margin like in AM-Softmax
    See Ref[1] for paper. And Ref[2] for paper from which I took the idea of using margin

    This implementation is different from the most open-source implementations in following ways:
    1) can work both on embeddings of size (bs, embedding_size) and on raw logits of size (bs x num_classes)
        in the second case logits are expected to be already scaled to unit sphere
    2) despite AdaCos being dynamic, still add an optional margin parameter
    3) calculate running average stats of B and θ, not batch-wise stats as in original paper

    Args:
        embedding_size (Optional[int, None]): if not None should be embedding size
        num_classes (Optional[int, None]): if not None should be number of output classes
        margin (float): margin in radians
        momentum (float): momentum for running average of B and θ. Recommended value is 0.3

    Input:
        y_pred (torch.Tensor): shape BS x N_classes of BS x Embedding_size
        y_true (torch.Tensor): one-hot encoded. shape BS x N_classes
    Reference:
        [1] Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations

    """

    def __init__(self, embedding_size=None, num_classes=None, margin=0.3, momentum=0.95):
        super().__init__()
        if embedding_size is None or num_classes is None:
            self.weight = None
        else:
            self.register_parameter("weight", nn.Parameter(torch.zeros(num_classes, embedding_size)))
        assert margin < (math.pi / 4), "need margin to be less than pi / 4 to avoid division by a very small value"
        self.margin = torch.tensor(margin)
        self.momentum = momentum
        self.prev_s = 10
        self.running_B = 1e5  # default value is chosen so that initial S is ~10
        self.running_theta = margin + 0.3
        self.eps = 1e-7
        self.final_criterion = 10
        self.idx = 0
        self.final_criterion = pt.losses.CrossEntropyLoss()

    def forward(self, features, y_true):
        if self.weight is not None:
            cosine = torch.nn.functional.linear(features, torch.nn.functional.normalize(self.weight))
        else:
            cosine = features
        self.margin = self.margin.to(features)  # cast dtype and device
        # cos_theta = cosine.clamp(-1 + self.eps, 1 - self.eps)
        # cos_theta = torch.cos(torch.acos(cos_theta + self.margin))
        if y_true.dim() == 1:
            y_true_one_hot = torch.zeros_like(cosine)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        else:
            y_true_one_hot = y_true.float()
            y_true = y_true_one_hot.argmax(-1).long()

        with torch.no_grad():
            B_batch = cosine[y_true_one_hot.eq(0)].mul(self.prev_s).exp().sum().div(y_true.size(0))
            # self.running_B = self.running_B * self.momentum + B_batch * (1 - self.momentum)
            # it's important to use median. mean wouldn't work!
            theta_batch = cosine.gather(dim=1, index=y_true[..., None]).median().clamp_min(self.margin + 0.3)
            # with cutmix lines below account for additional (small) class when calculating median. it's undesired
            # theta_batch = cosine[y_true_one_hot.ne(0)].median().clamp_min(self.margin + 0.3)
            # self.running_theta = self.running_theta * self.momentum + theta_batch * (1 - self.momentum)
            # using batch instead of running should help
            self.prev_s = B_batch.log() / (theta_batch - self.margin)

            # self.prev_s = self.running_B.log() / (self.running_theta - self.margin)
            self.prev_s = self.prev_s.clamp_max(25)

        cosine = cosine.scatter_add(dim=1, index=y_true[..., None], src=self.margin.expand(y_true.size(0), 1))

        self.idx += 1
        if self.idx % 100 == 0:
            print(
                f"\nRunning B: {self.running_B:.2f} Running theta: {self.running_theta:.2f} Running S: {self.prev_s:.2f} Batch theta: {theta_batch:.2f}"
            )
        return self.final_criterion(cosine * self.prev_s, y_true_one_hot)


class SphereMAELoss(Loss):
    # NOTE: This loss doesn't work if used alone, because it collapses
    # needs any additional loss to prevent it from collapse
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, cosine, y_true):
        """
        Args:
            features: already sphere normalized logits
            y_true: Class labels, not one-hot encoded
        """
        EPS = 1e-7
        theta = torch.acos(cosine.clamp(-1 + EPS, 1 - EPS))
        true_theta = theta.gather(dim=1, index=y_true.unsqueeze(-1))
        mask = true_theta.gt(self.threshold)
        if (~mask).all():  # if all samples are less than threshold need to avoid division by zero
            return true_theta.mul(0).sum()
        else:
            # minimize mean distance to true target vector for samples with angle less bigger than threshold
            return true_theta[mask].mean()


class SphereCosMAELoss(Loss):
    # NOTE: This loss doesn't work if used alone, because it collapses
    # needs any additional loss to prevent it from collapse
    # different from SphereMAELoss because it optimizes cosine not the actual angle
    # avoiding of arccos may help (or may not need to test)
    # default value 0.98 = arccos(0.2). don't push further than 0.2 rad
    def __init__(self, threshold=0.98):
        super().__init__()
        self.threshold = threshold

    def forward(self, cosine, y_true):
        """
        Args:
            cosine: already sphere normalized logits
            y_true: Class labels, not one-hot encoded
        """
        true_cosine = cosine.gather(dim=1, index=y_true.unsqueeze(-1))
        mask = true_cosine.lt(self.threshold)
        if (~mask).all():  # if all samples are greater than threshold need to avoid division by zero
            return true_cosine.mul(0).sum()
        else:
            # minimize cosine to true target vector (for samples with cosine lower than threshold)
            return 1 - true_cosine[mask].mean()


class NegativeContrastive(nn.Module):
    def __init__(self, eta=0.999):
        super().__init__()
        self.eta = eta

    # This loss maximizes inter-class distance by ensuring that examples of negative class are wide spread
    def forward(self, cosine, y_true):
        """
        Args:
            cosine: already sphere normalized logits
            y_true: Class labels, not one-hot encoded
        """
        # estimate max `s` as in LinCos-Softmax paper
        # in reality if batch_B drops lower than `num_classes` during training it's a good idea to decrease eta slightly
        s = np.log(self.eta / (1 - self.eta) * cosine.size(1))
        cosine_negative = cosine.scatter(dim=1, index=y_true.long().unsqueeze(-1), value=-1)
        loss = cosine_negative.mul(s).exp().sum(-1).add(1).log().mean()
        return loss


class DSoftmax_intra(nn.Module):
    def __init__(self, threshold=0.90):
        super().__init__()
        self.threshold = threshold

    # This loss maximizes inter-class distance by ensuring that examples of negative class are wide spread
    def forward(self, cosine, y_true):
        """
        Args:
            cosine: already sphere normalized logits
            y_true: Class labels, not one-hot encoded
        """
        true_cosine = cosine.gather(dim=1, index=y_true.unsqueeze(-1))
        with torch.no_grad():
            s = 16  # need large enough
            d_max = 0.9
            # lower bound makes sure loss properly converges at the beginning
            # upper bound makes sure loss properly converges at the end to 0 loss. Otherwise loss is always ~0.7
            # cos_median = true_cosine.detach().median().clamp(0.5, d_max)
            # min_loss = np.log(1 + np.e ** ((self.threshold - 1) * s))
            cos_median = self.threshold
        loss = (cos_median - true_cosine).mul(s).exp().add(1).log().mean()  # - min_loss
        # cosine_negative = cosine.scatter(dim=1, index=y_true.long().unsqueeze(-1), value=-1)
        # loss = cosine_negative.mul(s).exp().sum().add(1).log()
        return loss


class MyLoss1(Loss):
    def __init__(self, w_intra=1, w_inter=1, intra_threshold=0.9, eta=0.999):
        super().__init__()
        # self.loss_intra = DSoftmax_intra(intra_threshold)
        # self.loss_inter = NegativeContrastive()
        self.w_intra = w_intra
        self.w_inter = w_inter
        self.eta = eta
        self.intra_threshold = intra_threshold
        self.idx = 0
        self.mom = 0.99  # for debug printing only
        self.running_B = 0
        self.running_theta = 0
        self.running_neg_theta = 0

    def forward(self, cosine, y_true):
        if y_true.dim() == 1:
            y_true_one_hot = torch.zeros_like(cosine)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        else:
            y_true_one_hot = y_true.float()
            y_true = y_true_one_hot.argmax(-1).long()

        # inter class component aka Negative Contrastive
        s = np.log(self.eta / (1 - self.eta) * cosine.size(1))
        cosine_negative = cosine.where(
            y_true_one_hot.eq(0), torch.ones_like(cosine).neg()
        )  # fill positive class with -1
        l_inter = cosine_negative.mul(s).exp().sum(-1).add(1).log().mean()
        # TODO: remove top1 negative from loss computation. idea is that some classes are literali the same and this loss
        # penalizes even correct predictions strongly
        # cosine_negative = cosine.scatter(dim=1, index=y_true.long().unsqueeze(-1), value=-1)
        # l_inter = cosine_negative.mul(s).exp().sum(-1).add(1).log().mean()

        # inter class component aka D-Softmax intra
        # in case of softmax want only largest class to be close to 1, so use y_true
        true_cosine = cosine.gather(dim=1, index=y_true.unsqueeze(-1))
        s = 16  # need large enough
        l_intra = (self.intra_threshold - true_cosine).mul(s).exp().add(1).log().mean()

        # for debug
        with torch.no_grad():
            self.running_theta = self.running_theta * self.mom + true_cosine.median() * (1 - self.mom)
            B_batch = cosine_negative.mul(s).exp().sum(-1).mean()
            self.running_B = self.running_B * self.mom + B_batch * (1 - self.mom)
            self.running_neg_theta = self.running_neg_theta * self.mom + cosine_negative.median() * (1 - self.mom)

        if self.idx % 1000 == 0:
            # theta_batch = cosine.gather(dim=1, index=y_true[..., None]).median()
            # neg_cosine = cosine.scatter(dim=1, index=y_true.long().unsqueeze(-1), value=-1)
            # B_batch = neg_cosine.mul(13).exp().sum().div(y_true.size(0))
            logger.info(
                f"Batch B: {self.running_B:.2f} Pos theta median: {self.running_theta:.2f} Neg theta median: {self.running_neg_theta:.2f} L_intra: {l_intra:.2f} L_inter: {l_inter:.2f}"
            )
        self.idx += 1
        return l_intra * self.w_intra + l_inter * self.w_inter


class ArcCosSoftmax(pt.losses.CrossEntropyLoss):
    def forward(self, y_pred, y_true):
        EPS = 1e-7
        y_pred = -torch.acos(y_pred.clamp(-1 + EPS, 1 - EPS))
        return super().forward(y_pred, y_true)


# class AMSoftmax(pt.losses.CrossEntropyLoss):
#     def __init__(self, *args, margin=0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.margin = margin

#     def forward(self, y_pred, y_true):
#         cosine = torch.nn.functional.linear(features, torch.nn.functional.normalize(self.weight))
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size()).to(features)
#         one_hot.scatter_(1, y_true.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#             (1.0 - one_hot) * cosine
#         )  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s

#         EPS = 1e-7
#         y_pred = -torch.acos(y_pred.clamp(-1 + EPS, 1 - EPS))
#         return super().forward(y_pred, y_true)


class ArcCosSoftmaxCenter(pt.losses.CrossEntropyLoss):
    # combination of ArcCos + Center loss
    # ArcCos optimizes inter-class distance while Center Loss optimizes intra-class distance
    def __init__(self, *args, center_weight=1, **kwargs):
        self.center_weight = center_weight
        super().__init__(self, *args, **kwargs)

    def forward(self, y_pred, y_true):
        EPS = 1e-7
        theta = torch.acos(y_pred.clamp(-1 + EPS, 1 - EPS))
        cce_loss = super().forward(-theta, y_true)
        # if we're using cutmix only move to the largest target
        true_index = y_true[..., None] if y_true.dim() == 1 else y_true.argmax(-1, keepdim=True)
        # minimize MSE distance to true target vector (in radians). MSE needed to push further points stronger
        center_loss = theta.gather(dim=1, index=true_index).pow(2).mean()
        return cce_loss + self.center_weight * center_loss


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
}
