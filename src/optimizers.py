"""My version of optimizers. here we go again, aga"""

import math


import numpy as np

import torch
from typing import Any, Callable, Optional

from torch.optim import Optimizer
import torch.optim
from collections import defaultdict


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type).expand_as(x)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True).expand_as(x)


def unitwise_x_sq(x):
    """Computes mean of x ^ 2 layerwise"""
    if x.ndim <= 1:
        return x.pow(2).mean().expand_as(x)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.pow(2).mean(dim=tuple(range(1, x.ndim)), keepdim=True).expand_as(x)


class MyNovograd(Optimizer):
    r"""Fused re-implementation of Novograd. Should be close to apex.optimizers.FusedNovograd but in pure PyTorch

    Currently there are 2 implementations of Novograd I'm aware of 1. apex 2. torch-optimizers and both has it's flows

    This implementation doesn't save any memory compared to AdamW

    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional):
            learning rate (default: 1e-3)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional):
            term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional):
            weight decay coefficient (default: 1e-2)
        ema_norm_init (float, optional):
            value to use for exponential average norm initialization idea is to initialize with something non-zero.
            norm of first batch may be wrong approximation and harm the weight distribution during first couple of
            steps, so init with something large enough. default should be good enough, no need to tune

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=1e-2,
        ema_norm_init=1e-3,
        unitwise_norm=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, ema_norm_init=ema_norm_init)
        super().__init__(params, defaults)
        self.eps = eps  # eps
        self.unitwise_norm = unitwise_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            states = []
            ema_grad = []
            ema_norm = []
            params_with_grad = []

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("This optimizer does not support sparse gradients")

                    params_with_grad.append(p)
                    grads.append(p.grad)

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient norms. It should be scalar but this would cause problems
                    # with shape mismatch so init to tensor. This will require additional memory but I don't care for now
                    state["ema_norm"] = torch.full_like(p, group["ema_norm_init"], memory_format=torch.preserve_format)

                ema_grad.append(state["ema_grad"])
                ema_norm.append(state["ema_norm"])

                state["step"] += 1
                states.append(state)

            beta1, beta2 = group["betas"]

            # Decay the second moment
            if self.unitwise_norm:
                grad_norms = [unitwise_norm(w) for w in params_with_grad]
            else:
                # upd. in apex they average norm ^ 2 instead of norms
                grad_norms = [w.pow(2).sum().expand_as(w) for w in params_with_grad]
                # grad_norms = [w.norm(2).clamp_min(self.eps).expand_as(w) for w in params_with_grad]
            torch._foreach_mul_(ema_norm, beta2)
            torch._foreach_add_(ema_norm, grad_norms, alpha=1 - beta2)

            # Decay the first moment using normalized gradient. m = m * beta1 + grad / exp_grad_norm * (1 - beta1)
            torch._foreach_mul_(ema_grad, beta1)
            denom = torch._foreach_sqrt(ema_norm)
            torch._foreach_add_(denom, self.eps)
            # torch._foreach_addcdiv_(ema_grad, grads, denom, 1 - beta1)
            torch._foreach_add_(ema_grad, grads, alpha=1 - beta1)

            # if self.nesterov:
            #     exp_avg = torch._foreach_mul(exp_avg, beta1)  # not inplace here!
            #     torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            # p = p - lr * exp_avg
            # torch._foreach_add_(params_with_grad, ema_grad, alpha=-group["lr"])
            torch._foreach_addcdiv_(params_with_grad, ema_grad, denom, value=-group["lr"])

            # Decoupled weight decay (as in AdamW): p *= 1 - lr * wd
            torch._foreach_mul_(params_with_grad, 1 - group["lr"] * group["weight_decay"])

        return loss

    # TODO: refactor to a base class once foreach ops are in a good shape.
    def zero_grad(self, set_to_none: bool = False):
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)

                        if p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)

            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)


# copy-paste from apex code to validate to check that it works as expected
# it may be easier to modify this version than write from scratch
class NovogradApex(Optimizer):
    """
    Implements Novograd algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0),
        eps=1e-8,
        weight_decay=0,
        ema_norm_init=1e-3,
        unitwise_norm=False,
        wd_eps=None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)
        self.ema_norm_init = ema_norm_init
        self.unitwise_norm = unitwise_norm
        self.wd_eps = wd_eps

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.full_like(p.data, fill_value=self.ema_norm_init)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if self.unitwise_norm:
                    norm = unitwise_norm(grad)
                else:
                    norm = grad.pow(2).sum().expand_as(grad)  # expand to preserve shape

                # update second momentum
                exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # grad averaging like in Adam
                exp_avg.mul_(beta1)
                exp_avg.addcdiv_(grad, denom, value=1 - beta1)

                p.data.add_(exp_avg, alpha=-group["lr"])

                if self.wd_eps is None:
                    # Default decoupled weigh decay
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                else:
                    # weight decay only if |data| > eps
                    eps_data = p.data.abs().sub_(self.wd_eps).clamp_min_(0).mul_(p.data.sign())
                    p.data.sub_(eps_data, alpha=group["lr"] * group["weight_decay"])

        return loss


class AdamLayerwise(Optimizer):
    """
    Close to Adam but using layer-wise grad^2 instead of per-weight. Should let to lower adaptivity

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0),
        eps=1e-6,
        weight_decay=0,
        ema_norm_init=1e-3,
        weight_adapt=False,
        stable_wd=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)
        self.ema_norm_init = ema_norm_init
        self.weight_adapt = weight_adapt
        self.stable_wd = stable_wd

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.full_like(p.data, fill_value=self.ema_norm_init)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # update second momentum with layerwise grad ^ 2, not per-weight grad ^ 2 as in Adam
                # we can calculate grad ^ 2 unitwise but it leads to over-adaptivity and harms generalization
                grad_sq = grad.pow(2).mean().expand_as(grad)
                exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # grad is divided by running grad_sq first. It makes training more stable to outliers than Adam formulation
                exp_avg.mul_(beta1)
                exp_avg.addcdiv_(grad, denom, value=1 - beta1)

                step = exp_avg
                if self.weight_adapt:
                    # multiply by Root Mean Square of weights to avoid too large steps for small weights
                    weight_rms = p.data.pow(2).mean().sqrt().clamp_min_(1e-3)  # avoid weights being stacked at 0
                    step = step.mul(weight_rms)  # not inplace to avoid corrupting exp_avg!

                p.data.add_(step, alpha=-group["lr"])

                # if self.wd_eps is None:
                if self.stable_wd:
                    # produces None for some reasons
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] / denom)
                else:
                    # Default decoupled weigh decay
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                # else:
                # # weight decay only if |data| > eps
                # eps_data = p.data.abs().sub_(self.wd_eps).clamp_min_(0).mul_(p.data.sign())
                # p.data.sub_(eps_data, alpha=group["lr"] * group["weight_decay"])

        return loss


class MyAdai(Optimizer):
    """
    Close to Adam but using layer-wise grad^2 instead of per-weight. Should let to lower adaptivity

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.1, 0.99),
        eps=1e-3,
        weight_decay=0,
        ema_norm_init=1e-3,
        sgd_mom=False, # no beta3
        sqrt_mom=False, # take sqrt from both var estimates
        stable_wd=False, # 
        per_layer=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)
        self.ema_norm_init = ema_norm_init
        self.sgd_mom = sgd_mom
        self.sqrt_mom = sqrt_mom
        self.stable_wd = stable_wd
        self.per_layer = per_layer

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        exp_avg_sq_mean = self.ema_norm_init
        if len(self.state) != 0:
            if self.per_layer:
                exp_avg_sq_mean = sum(v["exp_avg_sq"] for v in self.state.values()) / len(self.state)
            else:
                exp_avg_sq_mean = sum(v["exp_avg_sq"].mean() for v in self.state.values()) / len(self.state)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    if self.per_layer:
                        state["exp_avg_sq"] = self.ema_norm_init
                    else:
                        state["exp_avg_sq"] = torch.full_like(p.data, self.ema_norm_init)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta0, beta2 = group["betas"]
                state["step"] += 1

                if self.per_layer:
                    grad_sq = grad.pow(2).mean().item()
                    exp_avg_sq = exp_avg_sq * beta2 + grad_sq * (1 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Motivation: update speed is proportional to grad, not grad^2
                # if grads for some layer are 10x smaller we want to move 10x faster. but in current implementation
                # we would move 100x faster which is strange and undesired. using sqrt fixes this
                if self.sqrt_mom:
                    beta1 = 1 - np.sqrt(exp_avg_sq / exp_avg_sq_mean) * beta0
                else:
                    beta1 = 1 - (exp_avg_sq / exp_avg_sq_mean) * beta0
                if self.per_layer:
                    beta1 = np.clip(beta1, 0, 1 - group["eps"])
                else:
                    beta1 = beta1.clamp(0, 1 - group["eps"])
                if self.sgd_mom:
                    exp_avg.mul_(beta1).add_(grad)  #
                else:
                    # No idea why authors use Adam formulation of momentum instead of SGD. In my opinion it kills all the benefits of adaptive intertia
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                p.data.add_(exp_avg, alpha=-group["lr"])

                if self.stable_wd:
                    # correction makes wd independent of current beta1. should be easier to tune
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] / (1 - beta1))
                else:
                    # Default decoupled weigh decay
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

        return loss


class AdaiS(Optimizer):
    r"""Implements Adai with stable/decoupled weight decay (AdaiS/AdaiW).
    It is based on
    `Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia`
    and
    `Stable Weight Decay Regularization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): beta0 and beta2 (default: (0.1, 0.99))
        eps (float, optional): the inertia bound (default: 1e-03)
        weight_decay (float, optional): weight decay (default: 0)

    """

    def __init__(self, params, lr=0, betas=(0.1, 0.99), eps=1e-3, weight_decay=0, ema_norm_init=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0]:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaiS, self).__init__(params, defaults)
        self.ema_norm_init = ema_norm_init

    def __setstate__(self, state):
        super(AdaiS, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_size += p.numel()
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.full_like(
                        p.data, self.ema_norm_init, memory_format=torch.preserve_format
                    )
                    # Cumulative products of beta1
                    state["beta1_prod"] = torch.ones_like(p.data, memory_format=torch.preserve_format)

                exp_avg_sq = state["exp_avg_sq"]
                beta0, beta2 = group["betas"]

                state["step"] += 1
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()

        # Calculate the mean of all elements in exp_avg_sq_hat
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Perform stable/decoupled weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                state = self.state[p]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta0, beta2 = group["betas"]
                beta1_prod = state["beta1_prod"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                beta1 = (1.0 - (exp_avg_sq_hat / exp_avg_sq_hat_mean).mul(beta0)).clamp(0.0, 1 - group["eps"])

                beta1_prod.mul_(beta1)
                bias_correction1 = 1 - beta1_prod

                # No idea why authors use Adam formulation of momentum instead of SGD. In my opinion it kills all the benefits of
                exp_avg.mul_(beta1).addcmul_(1 - beta1, grad)
                # exp_avg.mul_(beta1).addcmul_(1 - beta1, grad)

                exp_avg_hat = exp_avg.div(bias_correction1)

                step_size = group["lr"]
                p.data.add_(-step_size, exp_avg_hat)

        return loss


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.

    .. _MADGRAD: https://arxiv.org/abs/2101.11075

    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.

    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.

    On sparse problems both weight_decay and momentum should be set to 0.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    state["x0"] = torch.clone(p.data).detach()

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                x0 = state["x0"]

                # Accumulate second moments
                grad_sum_sq.addcmul_(grad, grad, value=lamb)
                rms = grad_sum_sq.pow(1 / 3).add_(eps)

                # Update s
                s.data.add_(grad, alpha=lamb)

                # Step
                z = x0.addcdiv(s, rms, value=-1)

                # p is a moving average of z
                # p = p * mom + z * (1 - mom)
                p.data.mul_(1 - ck).add_(z, alpha=ck)

                # Apply weight decay
                # grad.add_(p.data, alpha=decay) # original. not decoupled
                p.data.mul_(1 - decay)  # fixed. decoupled

        self.state["k"] += 1
        return loss
