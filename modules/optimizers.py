import math
import torch
from torch.optim import SGD, Adam, RMSprop
from torch.optim.optimizer import Optimizer, required

def optimizer_factory(optim_name):
    optim_name = optim_name.lower()
    if optim_name == 'sgd':
        return SGD
    elif optim_name == 'sgdw': 
        return SGDW
    elif optim_name == 'adam':
        return Adam
    elif optim_name =='adaw':
        return AdamW
    elif optim_name =='rmsprop':
        return RMSprop
    else:
        raise ValueError('Optimizer {} not found'.format(optim_name))

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                p.data.addcdiv_(-step_size, exp_avg, denom)
                

        return loss


class SGDW(Optimizer):
    r"""Implements SGDW algorithm.
    The SGDW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
       params (iterable): iterable of parameters to optimize or dicts defining
           parameter groups
       lr (float): learning rate
       momentum (float, optional): momentum factor (default: 0)
       weight_decay (float, optional): weight decay coefficient (default: 1e-2)
       dampening (float, optional): dampening for momentum (default: 0)
       nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
       >>> optimizer = torch.optim.SGDW(model.parameters(), lr=0.1, momentum=0.9)
       >>> optimizer.zero_grad()
       >>> loss_fn(model(input), target).backward()
       >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
       The implementation of SGD with Momentum/Nesterov subtly differs from
       Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                 v = \rho * v + g \\
                 p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
       velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
       other frameworks which employ an update of the form
        .. math::
            v = \rho * v + lr * g \\
            p = p - v
        The Nesterov version is analogously modified.
    .. _Decoupled Weight Decay Regularization:
       https://arxiv.org/abs/1711.05101
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
              if p.grad is None:
                  continue
              d_p = p.grad.data
              if momentum != 0:
                  param_state = self.state[p]
                  if 'momentum_buffer' not in param_state:
                      buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                  else:
                      buf = param_state['momentum_buffer']
                      buf.mul_(momentum).add_(1 - dampening, d_p)
                  if nesterov:
                      d_p = d_p.add(momentum, buf)
                  else:
                      d_p = buf
              # Apply weight decay. THE ONLY DIFFERENCE IS HERE
              if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
              # Apply momentum
              p.data.add_(-group['lr'], d_p)
        return loss