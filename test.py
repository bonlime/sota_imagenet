import torch
import torch.nn as nn

@torch.jit.script
def frn_train_forward(x, weight, bias, single_running_var, running_var, momentum: float, eps: float):
        x2_LN = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = x * (x2_LN + eps).rsqrt()
        with torch.no_grad():
            single_running_var.lerp_(x2_LN.mean(), 1 - momentum)
            r_LN = x2_LN.add(eps).div_(single_running_var).sqrt_().clamp_(1/5, 5)
        x = x * r_LN # Re-Normalization of LN, so that distribution is similar during training and testing

            # apply IN after LN to get diverse features
        x2_IN = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x = x * (x2_IN + eps).rsqrt()
        with torch.no_grad():
            # update running average with per-batch mean
            running_var.lerp_(x2_IN.mean(dim=0), 1 - momentum)
            r_IN = x2_IN.add(eps).div_(running_var).sqrt_().clamp_(1/5, 5)
        x = x * r_IN
        return x * weight + bias

@torch.jit.script
def frn_val_forward(x, weight, bias, single_running_var, running_var, eps: float) -> torch.Tensor:
    return x * (single_running_var + eps).rsqrt() * (running_var + eps).rsqrt() * weight + bias

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.95):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(1, num_features, 1, 1)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(1, num_features, 1, 1)))
        self.register_buffer("single_running_var", torch.ones(1))
        # it's called running var, but in fact this is running RMS
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.momentum = momentum
        self.eps = eps

    # v2
    # this version trains much worse
    def forward(self, x):
        if self.training:
            return frn_train_forward(x, self.weight, self.bias, self.single_running_var, self.running_var, self.momentum, self.eps)
        else:
            return frn_val_forward(x, self.weight, self.bias, self.single_running_var, self.running_var, self.eps)



inp = torch.rand(4, 16, 32, 32).cuda()
frn = FRN(16).cuda()

for _ in range(10):
    frn(inp)

print("Finished")