import torch

import pytorch_tools as pt
from pytorch_tools.fit_wrapper.callbacks import Callback


class WeightDistributionTB(Callback):
    """Plot weight distribution for each epoch to TB"""

    def on_epoch_begin(self):
        for n, p in self.state.model.state_dict().items():
            self.state.tb_logger.add_histogram(f"model/{n}", p.flatten(), self.state.global_sample_step)


class WeightNorm(Callback):
    """init weights properly and make sure they are normalized during training.
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

