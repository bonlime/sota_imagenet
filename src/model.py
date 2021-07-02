"""temporary place for experiments with model. when it's mature will move this to pytorch_tools"""


"""c_model is model with Caffe-like explicit model constructor. While it makes configs
slightly larger, it also allows much greater flexibility than using separate class for each model
"""

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import List, Dict, Union


import torch
import torch.nn as nn

import pytorch_tools as pt
from pytorch_tools.utils.misc import listify
from pytorch_tools.modules.residual import conv1x1, conv3x3, DropConnect
from pytorch_tools.modules import ABN


## some blocks defintion

class PreBasicBlock(nn.Module):
    """BasicBlock with preactivatoin & without downsample support"""

    def __init__(
        self, in_chs, out_chs, mid_chs=None, groups=1, groups_width=None, norm_layer=ABN, norm_act="relu", keep_prob=1,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        mid_chs = mid_chs or out_chs
        groups = in_chs // groups_width if groups_width else groups
        layers = [
            ("bn1", norm_layer(in_chs, activation=norm_act)),
            ("conv1", conv3x3(in_chs, mid_chs)),
            ("bn2", norm_layer(mid_chs, activation=norm_act)),
            ("conv2", conv3x3(mid_chs, out_chs)),
            ("droppath", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity())
        ]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.block(x)
        if self.in_chs == self.out_chs:
            out += x
        else:
            out[:, :self.in_chs] += x
        return out

class PreInvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        mid_chs=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        mid_chs = mid_chs or out_chs
        layers = [
            ("bn1", norm_layer(in_chs, activation=norm_act)),
            ("conv_pw", conv1x1(in_chs, mid_chs)),
            ("bn2", norm_layer(mid_chs, activation=norm_act)),
            ("conv_dw", conv3x3(mid_chs, mid_chs, groups=mid_chs)),
            ("bn3", norm_layer(mid_chs, activation=norm_act)),
            ("conv_pw2", conv1x1(mid_chs, out_chs)),
            ("droppath", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity())
        ]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.block(x)
        if self.in_chs == self.out_chs:
            out += x
        else:
            out[:, :self.in_chs] += x
        return out


@dataclass
class LayerDef:
    # previous layer
    prev_l: int
    # number of repeats
    n: int
    # which layer
    module: Union[str, nn.Module]
    args: List = field(default_factory=lambda: tuple())
    kwargs: Dict = field(default_factory=lambda: dict())


class Concat(nn.Module):
    def forward(self, *args):
        return torch.cat(*args, dim=1)


class CModel(nn.Module):
    """
    Args:
        features_idx (List[int]):
            from which layers to additionally save features
    """

    def __init__(
        self, layer_config: List[LayerDef], extra_kwargs: Dict[str, Dict] = None, features_idx: List[int] = None,
    ):
        super().__init__()
        if not isinstance(layer_config[0], LayerDef):
            layer_config = [LayerDef(*l) for l in layer_config]
        if extra_kwargs is not None:
            self._update_config_with_extra_params(layer_config, extra_kwargs)
        layers, saved = self._parse_config(layer_config)
        self.layers: nn.Module = layers
        self.saved: List[int] = saved

    @staticmethod
    def _update_config_with_extra_params(layer_config: List[LayerDef], extra_kwargs: Dict[str, Dict]):
        for l_name, l_kwargs in extra_kwargs.items():
            for l in layer_config:
                if l.module == l_name:
                    l.kwargs.update(**l_kwargs)
        
    @staticmethod
    def _parse_config(layer_config: List[LayerDef]):
        saved = []
        layers = []
        for l_idx, l in enumerate(layer_config):
            l.module = eval(l.module) if isinstance(l.module, str) else l.module  # eval strings
            l.args = [eval(i) if isinstance(i, str) else i for i in listify(l.args)]
            l.kwargs = {k: (eval(v) if isinstance(v, str) else v) for k, v in l.kwargs.items()}

            if l.n == 1:
                m = l.module(*l.args, **l.kwargs)
            else:
                m = nn.Sequential(*[l.module(*l.args, **l.kwargs) for _ in range(l.n)])
            # add some information about from/idx
            m.prev_l = l.prev_l
            m.idx = l_idx
            layers.append(m)
            saved.extend(l_idx + i for i in listify(l.prev_l) if i != -1)

        return nn.ModuleList(layers), saved

    def forward(self, x):
        saved_outputs: List[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer.prev_l, list):
                x = [x if j == -1 else saved_outputs[j] for j in layer.prev_l]
            elif layer.prev_l != -1:
                x = saved_outputs[layer.prev_l]

            # print('inp', [i.shape for i in listify(x)])
            # print('saved', [i.shape for i in listify(saved_outputs) if i is not None])

            x = layer(x)
            saved_outputs.append(x if layer.idx in self.saved else None)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        valid_weights = []
        for key, value in state_dict.items():
            if "num_batches_tracked" in key:
                continue
            valid_weights.append(value)
        new_sd = OrderedDict(zip(self.state_dict().keys(), valid_weights))
        super().load_state_dict(new_sd, **kwargs)


# layer_config = [
#     (-1, 1, 'pt.modules.SpaceToDepth', (2,)), # 0
#     (-1, 1, 'conv3x3', (12, 32, 2)), # 1
#     (-1, 1, conv3x3, (32, 64), {'bias': True}), # 2
#     (-2, 1, conv3x3, (32, 96)), # 3
#     ([-1, -2], 1, 'Concat'), # 4 [3, 2]
# ]


# fully matches R34
# layer_config = [
#     (-1, 1, 'Conv2d', (3, 64, 7, 2, 3), dict(bias=False)), # 0
#     (-1, 1, 'ABN', 64),
#     (-1, 1, 'torch.nn.MaxPool2d', (3, 2, 1)),
#     (-1, 3, 'pt.modules.BasicBlock', (64, 64)),
#     (-1, 1, 'pt.modules.BasicBlock', (64, 128), dict(stride=2, downsample='nn.Sequential(conv1x1(64, 128, 2), ABN(128))')),
#     (-1, 3, 'pt.modules.BasicBlock', (128, 128)),
#     (-1, 1, 'pt.modules.BasicBlock', (128, 256), dict(stride=2, downsample='nn.Sequential(conv1x1(128, 256, 2), ABN(256))')),
#     (-1, 5, 'pt.modules.BasicBlock', (256, 256)),
#     (-1, 1, 'pt.modules.BasicBlock', (256, 512), dict(stride=2, downsample='nn.Sequential(conv1x1(256, 512, 2), ABN(512))')),
#     (-1, 2, 'pt.modules.BasicBlock', (512, 512)),
#     (-1, 1, 'pt.modules.FastGlobalAvgPool2d', (), dict(flatten=True)),
#     (-1, 1, 'nn.Dropout', 0.0),
#     (-1, 1, 'nn.Linear', (512, 1000)),
# ]

# fully matches R50
# layer_config = [
#     (-1, 1, 'Conv2d', (3, 64, 7, 2, 3), dict(bias=False)), # 0
#     (-1, 1, 'ABN', 64, dict(activation="'relu'")),
#     (-1, 1, 'torch.nn.MaxPool2d', (3, 2, 1)),
#     (-1, 1, 'pt.modules.Bottleneck', (64, 64), dict(downsample="nn.Sequential(conv1x1(64, 256), ABN(256, activation='identity'))")),
#     (-1, 2, 'pt.modules.Bottleneck', (256, 64)),
#     (-1, 1, 'pt.modules.Bottleneck', (256, 128), dict(stride=2, downsample="nn.Sequential(conv1x1(256, 512, 2), ABN(512, activation='identity'))")),
#     (-1, 3, 'pt.modules.Bottleneck', (512, 128)),
#     (-1, 1, 'pt.modules.Bottleneck', (512, 256), dict(stride=2, downsample="nn.Sequential(conv1x1(512, 1024, 2), ABN(1024, activation='identity'))")),
#     (-1, 5, 'pt.modules.Bottleneck', (1024, 256)),
#     (-1, 1, 'pt.modules.Bottleneck', (1024, 512), dict(stride=2, downsample="nn.Sequential(conv1x1(1024, 2048, 2), ABN(2048, activation='identity'))")),
#     (-1, 2, 'pt.modules.Bottleneck', (2048, 512)),
#     (-1, 1, 'pt.modules.FastGlobalAvgPool2d', (), dict(flatten=True)),
#     (-1, 1, 'nn.Dropout', (), dict(p=0, inplace=False)),
#     (-1, 1, 'nn.Linear', (2048, 1000)),
# ]

# layer_config = [
#     (-1, 1, 'pt.modules.SpaceToDepth', 2), # 0
#     (-1, 1, 'conv3x3', (12, 32)), # 0
#     (-1, 1, 'pt.modules.residual.RepVGGBlock', (32, 64, 1)),
#     (-1, 3, 'pt.modules.residual.RepVGGBlock', (64, 64, 1)),
#     (-1, 1, 'pt.modules.residual.RepVGGBlock', (64, 96, 1)),
#     (-1, 1, 'pt.modules.BlurPool', 96),
#     (-1, 6, 'pt.modules.residual.RepVGGBlock', (96, 96, 1)),
# ]

# like BNet
layer_config = [
    (-1, 1, "pt.modules.SpaceToDepth", 2),  # 0
    (-1, 1, "conv3x3", (12, 64)),  # 0
    (-1, 1, "pt.modules.BlurPool", 64),
    (-1, 6, "pt.modules.residual.FusedRepVGGBlock", (64, 64)),
    (-1, 1, "pt.modules.BlurPool", 64),
    (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (64, 128)),
    (-1, 7, "pt.modules.residual.FusedRepVGGBlock", (128, 128)),
    (-1, 1, "pt.modules.BlurPool", 128),
    (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (128, 256)),
    (-1, 11, "pt.modules.residual.FusedRepVGGBlock", (256, 256)),
    (-1, 1, "pt.modules.BlurPool", 256),
    (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (256, 512)),
    (-1, 5, "pt.modules.residual.FusedRepVGGBlock", (512, 512)),
    (-1, 1, "pt.modules.FastGlobalAvgPool2d", (), dict(flatten=True)),
    (-1, 1, "nn.Dropout", (), dict(p=0, inplace=False)),
    (-1, 1, "nn.Linear", (512, 1000)),
]

#     (-1, 1, 'ABN', 64, dict(activation="'relu'")),
#     (-1, 1, 'torch.nn.MaxPool2d', (3, 2, 1)),
#     (-1, 1, 'pt.modules.Bottleneck', (64, 64), dict(downsample="nn.Sequential(conv1x1(64, 256), ABN(256, activation='identity'))")),
#     (-1, 2, 'pt.modules.Bottleneck', (256, 64)),
#     (-1, 1, 'pt.modules.Bottleneck', (256, 128), dict(stride=2, downsample="nn.Sequential(conv1x1(256, 512, 2), ABN(512, activation='identity'))")),
#     (-1, 3, 'pt.modules.Bottleneck', (512, 128)),
#     (-1, 1, 'pt.modules.Bottleneck', (512, 256), dict(stride=2, downsample="nn.Sequential(conv1x1(512, 1024, 2), ABN(1024, activation='identity'))")),
#     (-1, 5, 'pt.modules.Bottleneck', (1024, 256)),
#     (-1, 1, 'pt.modules.Bottleneck', (1024, 512), dict(stride=2, downsample="nn.Sequential(conv1x1(1024, 2048, 2), ABN(2048, activation='identity'))")),
#     (-1, 2, 'pt.modules.Bottleneck', (2048, 512)),
#     (-1, 1, 'pt.modules.FastGlobalAvgPool2d', (), dict(flatten=True)),
#     (-1, 1, 'nn.Dropout', (), dict(p=0, inplace=False)),
#     (-1, 1, 'nn.Linear', (2048, 1000)),


model = CModel(layer_config).cuda()
# print(model)
inp = torch.rand(16, 3, 224, 224).cuda()

model(inp).shape
