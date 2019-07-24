import torch

# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
def bnwd_optim_params(model):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum
    bn_params = get_bn_params(model)
    rem_params = [p for p in model.parameters() if p not in bn_params and p.requires_grad]
    return [{'params':bn_params,'weight_decay':0}, {'params':rem_params}]