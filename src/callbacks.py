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


@pt_clb.rank_zero_only
class SpectralDistributionTB(pt_clb.Callback):
    """Plot spectrum of convolutional / FC layers each epoch to TB"""

    def on_epoch_begin(self):
        for n, p in self.state.model.state_dict().items():
            if hasattr(p, "weight"):
                spectrum = torch.linalg.svdvals(p.weight.view(p.weight.size(0), -1).detach())
            self.state.tb_logger.add_histogram(f"spectrum/{n}", spectrum, self.state.global_sample_step)

@pt_clb.rank_zero_only
class GradDistributionTB(pt_clb.Callback):
    """Log distribution of square of gradients during training"""
    def __init__(self, log_every=500, subsample=10, state_keys=['exp_avg', 'exp_avg_sq']):
        self.log_every = log_every
        self.subsample = subsample
        self.state_keys = state_keys

    def on_batch_end(self):
        if self.state.step % self.log_every != 0:
            return

        # logger.info("Logging distribution")
        # log distribution for optimizer state
        for state_k in self.state_keys:
            all_gathered = []
            for v in self.state.optimizer.state.values():
                all_gathered.append(v[state_k].flatten().abs().sort().values[::self.subsample])
            all_gathered = torch.cat(all_gathered)
            # clamp_min to avoid outliers distorting the chart too much
            all_gathered_log = all_gathered.sort().values[::self.subsample].log10().clamp_min(-15)
            self.state.tb_logger.add_histogram(f"optim/{state_k}_log", all_gathered_log, self.state.global_sample_step)

        # log distribution for all model weights combined
        all_gathered = []
        for p in self.state.model.parameters():
            all_gathered.append(p.flatten().abs().sort().values[::self.subsample])
        all_gathered = torch.cat(all_gathered)
        # clamp_min to avoid outliers distorting the chart too much
        all_gathered_log = all_gathered.sort().values[::self.subsample].log10().clamp_min(-15)
        self.state.tb_logger.add_histogram(f"optim/model_params_log", all_gathered_log, self.state.global_sample_step)

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

    NOTE: model should be initialized using kaiming or orthogonal init for this callback to work correctly
    """

    def on_begin(self):
        logger.info("Using backward Centered Weight Normalization for weights")

    @torch.no_grad()
    def on_batch_end(self):
        for m in self.state.model.modules():
            # second check is needed to filter ECA attention weights. FIXME: remove this hack
            if not hasattr(m, "weight") or m.weight.numel() < 64:
                continue
            w = m.weight.view(m.weight.size(0), -1) # chs_out x chs_in x ks x ks -> chs_out x (chs_in * ks * ks)
            w -= w.mean(dim=-1, keepdim=True) # helps to prevent mean shift
            w /= w.norm(dim=-1, keepdim=True) # normalize to unit sphere
            m.weight.data.copy_(w.view_as(m.weight))


class OrthoLoss(pt.losses.Loss):
    # after many experiments this loss still doesn't work :( not sure why
    def __init__(self, model, eps=1e-2, min_filters=384, min_norm=1, debug=False):
        super().__init__()
        self.model = model
        self.eps = eps
        self.min_filters = min_filters
        self.min_norm = min_norm
        self.debug = debug

    def forward(self, *args):
        loss = 0
        for n, m in self.model.named_modules():
            if not isinstance(m, torch.nn.Conv2d):
                continue
            weight = m.parametrizations.weight.original if hasattr(m, "parametrizations") else m.weight
            mat = weight.view(weight.size(0), -1)
            # avoid asking for impossible (finding orthonormal basis larger than space)
            if mat.size(0) > mat.size(1) or mat.size(0) < self.min_filters:
                # print(n)
                continue  # skip when increasing number of channels
            # in original implementation it's mat.T @ mat but I think it should be mat @ mat.T to show correlation between different filters
            # originally I wrote it this way. but normalization prevents orthonormality and harms training
            # corr = mat @ mat.T / (mat.norm(dim=-1).pow(2) + self.eps) - torch.eye(mat.size(0), device=mat.device)
            corr = mat @ mat.T - torch.eye(mat.size(0), device=mat.device)
            corr_norm = corr.norm() / corr.size(0) # want to normalize to avoid larger filters being penalized more
            if corr_norm > self.min_norm:
                loss += corr.norm()
            if self.debug:
                print(f"{n:40s}: {mat.shape} {corr.norm().item():.3f}")
        return loss


class OrthoLoss2(pt.losses.Loss):
    # after carefull rereading "Orthogonal Convolutional Neural Networks" I've realised previous implementation was wrong
    # this is correct implementation
    def __init__(self, model, eps=1e-4):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, *args):
        total_loss = 0
        for n, m in self.model.named_modules():
            if not isinstance(m, torch.nn.Conv2d) or m.stride == 2:
                continue
            # if using spectral norm, regularize original weights to avoid gradient computation errors
            # maybe it's possible to apply loss to weighs after spectral norm but I don't know how :(
            weight = m.parametrizations.weight.original if hasattr(m, "parametrizations") else m.weight
            mat = weight.view(weight.size(0), -1)
            # avoid asking for impossible (finding orthonormal basis larger than space)
            if mat.size(0) > mat.size(1):
                continue
            corr = torch.conv2d(weight, weight, padding=weight.size(2) - 1)
            # make sure self-correlation for all kernels is exactly 1
            corr = corr / (mat.norm(dim=-1, keepdim=True).pow(2).view(mat.size(0), 1, 1) + self.eps)
            # chs_out x chs_out x 5 x 5 (for 3x3 convolution)
            target = torch.zeros_like(corr)
            target[:, :, target.size(2) // 2, target.size(2) // 2] = 1
            loss = (corr - target).norm()
            # print(f"{n:40s}: {mat.shape} {loss.item():.3f}")
            total_loss += loss
        return total_loss


class OrthoLossClb(pt_clb.Callback):
    def __init__(self, weight=0.01, type=1, **kwargs):
        self.weight = weight
        self.kwargs = kwargs
        self.type = type

    def on_begin(self):
        if self.type == 1:
            # kernel orthogonalization
            self.state.criterion += OrthoLoss(self.state.model, **self.kwargs) * self.weight
        elif self.type == 2:
            # convolutional orthogonalization
            self.state.criterion += OrthoLoss2(self.state.model, **self.kwargs) * self.weight


class NormLoss(pt.losses.Loss):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        total_loss = 0
        for n, m in self.model.named_modules():
            # second check is needed to filter ECA attention weights. FIXME: remove this hack
            if not hasattr(m, "weight") or m.weight.numel() < 64:
                continue
            mat = m.weight.view(m.weight.size(0), -1)
            loss = (1 - mat.norm(dim=-1)).pow(2).mean()
            print(f"{n:40s}: {mat.shape} {loss.item():.3f}")
            total_loss += loss
        return total_loss


class NormLossClb(pt_clb.Callback):
    def __init__(self, weight=1e-4):
        self.weight = weight

    def on_begin(self):
        self.state.criterion += NormLoss(self.state.model) * self.weight


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


class OrthoInitClb(pt_clb.Callback):
    def __init__(self, gain=1):
        self.gain = gain
        # if using scheduler `on_begin` would be called multiple times
        self.has_been_init = False

    def on_begin(self):
        if self.has_been_init:
            return
        logger.info("Applying orthogonal initialization")
        for m in self.state.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=self.gain)
        self.has_been_init = True


def unitwise_norm(x, norm_type=2.0, expand_as=False):
    if x.ndim <= 1:
        res = x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        res = x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)
    return res.expand_as(x) if expand_as else res


class SAMOriginal(pt_clb.Callback):
    """Very close to original implementation of ASAM from SamsungLabs"""

    def __init__(self, rho=0.5, eta=0.01):
        self.rho = rho
        self.eta = eta


    @torch.no_grad()
    def on_after_backward(self):
        # skip first step, to allow optimizer to initialize it's own state
        if len(self.state.optimizer.state) == 0:
            return
        
        # first backward has already been computed (with grad scaled loss)
        # need to calculate epsilon and backward using new weight
        self.state.grad_scaler.unscale_(self.state.optimizer)

        scale = self.rho / self._grad_norm()
        for group in self.state.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                eps = p.pow(2).clamp_min_(self.eta) * p.grad * scale if p.ndim > 1 else p.grad * scale
                # store eps_step in optimizer to avoid having own state_dict
                self.state.optimizer.state[p]['eps_step'] = eps
                p.add_(eps)
                
        # without update scaler thinks grads are unscaled (because we explicitly called `unscale_`). this leads to NaNs
        self.state.grad_scaler.update()
        self.state.optimizer.zero_grad()
        #         __
        # compute \/ L (w + eps)
        with torch.enable_grad():
            with torch.cuda.amp.autocast():
                data, target = self.state.input
                loss_second = self.state.criterion(self.state.model(data), target)
            self.state.grad_scaler.scale(loss_second).backward()

        for group in self.state.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state.optimizer.state[p]['eps_step'])
        # after that default optimizer would perform its' step as usual

    @torch.no_grad()
    def _grad_norm(self):
        wgrads = []
        for group in self.state.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # original implementation uses norm from all network and 
                # using p.ndim > 1 instead `if 'weight' in name`, it have the same meaning
                pw = p.grad * p.abs().clamp_min_(self.eta) if p.ndim > 1 else p.grad
                wgrads.append(pw.norm())
        return torch.stack(wgrads).norm().clamp_min(2e-5)
class SAM(pt_clb.Callback):
    """
    Implements Sharpness-aware minimization [1] as Callback.

    There few implementation choices which are different from existing open-source implementations
    1) grad norm is computed for each layer separately. In my understanding this is usefull to unit normalize gradient for each layer
        while taking global norm doesn't have a clear explanation

    2) Adaptive version of SAM (ASAM [2]) is implemented by multiplying epsilon for each layer with corresponding weight using unit-wise norm.
        The motivation is similar to Adaptive Gradient Clipping [3], we want flat minimum in ball of fixed

    in paper about AGC they mention that they were able to train using LARS which ignores magnitude of gradient completely, but this lead to degrade in performance
    this could be interpreted as

    Args:
        unitwise (bool):
            if True epsilon is scaled unit-wise separately for each element in the gradient
            if False it's scaled using weight norm and grad norm over the whole layer

    Ref:
        [1] SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY IMPROVING GENERALIZATION
        [2] ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks
        [3] High-Performance Large-Scale Image Recognition Without Normalization
    """

    def __init__(self, unitwise=False, rho=0.01):
        self.unitwise = unitwise
        self.rho = rho
        self.eps = 1e-5  # for numerical stability
        self.eps_2 = 1e-3  # defines minimum radius in which want flat minimum

    @torch.no_grad()
    def on_after_backward(self):
        # first backward has already been computed (with grad scaled loss)
        # need to calculate epsilon and backward using new weight
        self.state.grad_scaler.unscale_(self.state.optimizer)

        # lists for aggregation are outside loop to avoid the need for looping 2nd time
        params_with_grad = []
        grads = []
        for group in self.state.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.clone())
                    params_with_grad.append(p)

        if self.unitwise:
            grad_norms = [unitwise_norm(w, expand_as=True).clamp_min(self.eps) for w in grads]
            weight_norms = [unitwise_norm(w, expand_as=True).clamp_min(self.eps_2) for w in params_with_grad]
        else:
            grad_norms = [w.norm(2).clamp_min(self.eps).expand_as(w) for w in grads]
            # or eps_2?
            weight_norms = [w.norm(2).clamp_min(self.eps_2).expand_as(w) for w in params_with_grad]

        # epsilon = || w || / ||g|| * g * rho
        # epsilon = weight_norms # rename for convenience
        epsilon = torch._foreach_div(weight_norms, grad_norms)
        torch._foreach_mul_(epsilon, grads)
        torch._foreach_mul_(epsilon, self.rho)

        # variant 2
        # epsilon = grads # rename for convenience
        # torch._foreach_div_(grads, grad_norms)

        # virtual step toward epsilon
        torch._foreach_add_(params_with_grad, epsilon)

        # without update scaler thinks grads are unscaled (because we explicitly called `unscale_`). this leads to NaNs
        self.state.grad_scaler.update()
        self.state.optimizer.zero_grad()
        #         __
        # compute \/ L (w + eps)
        with torch.enable_grad():
            with torch.cuda.amp.autocast():
                data, target = self.state.input
                loss_second = self.state.criterion(self.state.model(data), target)
            self.state.grad_scaler.scale(loss_second).backward()

        # virtual step back to the original point
        torch._foreach_sub_(params_with_grad, epsilon)

        # after that default optimizer would perform its' step as usual