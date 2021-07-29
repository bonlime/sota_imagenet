import torch
import numpy as np
from loguru import logger
from functools import partial

import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from torch._C import device
from torch.nn.modules.conv import Conv2d



@pt_clb.rank_zero_only
class WeightDistributionTB(pt_clb.Callback):
    """Plot weight distribution for each epoch to TB"""

    def on_epoch_begin(self):
        for n, p in self.state.model.state_dict().items():
            self.state.tb_logger.add_histogram(f"model/{n}", p.flatten(), self.state.global_sample_step)


class ForwardWeightNorm(pt_clb.Callback):
    """Turn convs into StdConvs this is different from WeightNorm which implements the same idea but in backward mode
    `torch.nn.utils.parametrize` requires torch > 1.9.0
    """

    def __init__(self, gamma: float = None, use_std: bool = False):
        if use_std:
            assert gamma is not None, "You have to pass gamma to allow chaning std of weights"
            self.func = partial(pt.utils.misc.normalize_conv_weight, gamma=gamma)
        else:
            self.func = pt.utils.misc.zero_mean_conv_weight

    def on_begin(self):
        for m in self.state.model.modules():
            # turn to ScaledStdConv only usual (not DW) convs
            if isinstance(m, torch.nn.Conv2d) and m.groups == 1:
                torch.nn.utils.parametrize.register_parametrization(m, "weight", self.func)

    def on_end(self):
        # remove parametrization. the weight's would be converted to parametrized version automatically
        for m in self.state.model.modules():
            if torch.nn.utils.parametrize.is_parametrized(m):
                torch.nn.utils.parametrize.remove_parametrizations(m, "weight")

class ForwardSpectralNorm(pt_clb.Callback):
    """Apply spectral norm to all convs `torch.nn.utils.parametrize` requires torch > 1.9.0"""

    def on_begin(self):
        logger.info("Adding spectral norm parametrization for weights")
        # TODO: maybe add kaiming normal init here? or it shouldn't matter
        for m in self.state.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.parametrizations.spectral_norm(m)

    def on_end(self):
        # remove parametrization. the weight's would be converted to parametrized version automatically
        for m in self.state.model.modules():
            if torch.nn.utils.parametrize.is_parametrized(m):
                torch.nn.utils.parametrize.remove_parametrizations(m, "weight")


class WeightNorm(pt_clb.Callback):
    """make sure weights are normalized during training.
    This implementation is different from the one in the literature and performs so called `backward scaled weight normalization`

    gamma:
        gain from nonlin. see NFNet paper for details
    skip_dw:
        flag to avoid normalizing dw convs because it limits their representation power significantly

    NOTE: model should be initialized using kaiming init for this callback to work correctly
    """

    def __init__(self, gamma: float = None, use_std: bool = False):
        self.gamma = gamma
        self.use_std = use_std
        if use_std:
            assert gamma is not None, "You have to pass gamma to allow chaning std of weights"

    @torch.no_grad()
    def on_batch_end(self):
        for m in self.state.model.modules():
            if isinstance(m, torch.nn.Conv2d) and m.groups == 1:
                if self.use_std:
                    m.weight.data.copy_(pt.utils.misc.normalize_conv_weight(m.weight, gamma=self.gamma))
                else:
                    pt.utils.misc.zero_mean_conv_weight(m.weight)  # it's inplace

class OrthoLoss(pt.losses.Loss):
    def __init__(self, model, eps=1e-2):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, *args):
        loss = 0
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                mat = m.weight.view(m.weight.size(0), -1)
                # avoid asking for impossible (finding orthonormal basis larger than space)
                if mat.size(0) > mat.size(1):
                    mat = mat.T
                # in original implementation it's mat.T @ mat but I think it should be mat @ mat.T to show correlation between different filters
                corr = mat @ mat.T / (mat.norm(dim=-1).pow(2) + self.eps) - torch.eye(mat.size(0), device=mat.device)
                loss += corr.norm()
        return loss

class OrthoLossClb(pt_clb.Callback):
    def __init__(self, weight=0.01):
        self.weight = weight
    
    def on_begin(self):
        self.state.criterion += OrthoLoss(self.state.model) * self.weight

class CutmixMixup(pt_clb.Cutmix, pt_clb.Mixup):
    """combines CutMix and Mixup and applyes one or another randomly"""

    def __init__(self, cutmix_alpha, mixup_alpha, prob=0.5):
        self.cutmix_tb = torch.distributions.Beta(cutmix_alpha, cutmix_alpha)
        self.mixup_tb = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        self.prob = prob 
        self.prev_input = None

    def on_batch_begin(self):
        if np.random.rand() > 0.5:
            self.tb = self.cutmix_tb
            self.state.input = self.cutmix(*self.state.input)
        else:
            self.tb = self.mixup_tb
            self.state.input = self.mixup(*self.state.input)
