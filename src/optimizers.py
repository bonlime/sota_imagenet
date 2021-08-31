"""My version of optimizers. here we go again, aga"""

import torch
from loguru import logger
from torch._C import dtype, memory_format
from torch.optim import Optimizer
from collections import defaultdict


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type).expand_as(x)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True).expand_as(x)


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
        eps=1e-6,
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
        self.eps = eps # eps
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
                grad_norms = [w.norm(2).clamp_min(self.eps).expand_as(w) for w in params_with_grad]
            torch._foreach_mul_(ema_norm, beta2)
            torch._foreach_add_(ema_norm, grad_norms, alpha=1 - beta2)

            # Decay the first moment using normalized gradient. m = m * beta1 + grad / exp_grad_norm * (1 - beta1)
            torch._foreach_mul_(ema_grad, beta1)
            torch._foreach_addcdiv_(ema_grad, grads, ema_norm, 1 - beta1)

            # if self.nesterov:
            #     exp_avg = torch._foreach_mul(exp_avg, beta1)  # not inplace here!
            #     torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            # p = p - lr * exp_avg
            torch._foreach_add_(params_with_grad, ema_grad, alpha=-group["lr"])

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


