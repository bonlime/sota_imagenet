"""temporary place for experiments with model. when it's mature will move this to pytorch_tools"""


"""c_model is model with Caffe-like explicit model constructor. While it makes configs
slightly larger, it also allows much greater flexibility than using separate class for each model
"""

from loguru import logger
from copy import deepcopy
from collections import OrderedDict
from typing import List, Dict, Union
from dataclasses import dataclass, field
from pytorch_tools.modules.pooling import BlurPool


import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_tools as pt
from pytorch_tools.utils.misc import listify
from pytorch_tools.modules.residual import XCA, conv1x1, conv3x3, DropConnect
from pytorch_tools.modules import ABN
from pytorch_tools.modules import activation_from_name
from pytorch_tools.modules.residual import get_attn, XCA


## some blocks defintion


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.
    Args:
        gamma (float):
        gain_init (float):
        eps (float):
    Ref:
        [1] Characterizing signal propagation to close the performance gap in unnormalized ResNets (https://arxiv.org/abs/2101.08692)"""

    def __init__(
        self,
        in_chs,
        out_chs,
        *args,
        gamma=1.0,
        gain_init=1.0,
        eps=1e-6,
        n_heads=1,
        norm=False,
        partial_conv=False,
        coord_conv=False,
        **kwargs,
    ):
        out_chs *= n_heads
        if coord_conv:
            in_chs += 2
        super().__init__(in_chs, out_chs, *args, **kwargs)
        # gamma * 1 / sqrt(fan-in). multiply by num heads to compensate for mean
        self.scale = gamma * self.weight[0].numel() ** -0.5 * n_heads ** 0.5
        if gain_init is not None:
            self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), float(gain_init)))
        else:
            self.register_buffer("gain", torch.ones(out_chs, 1, 1, 1)) # just a scalar
        self.gamma = gamma
        self.eps = eps
        self.n_heads = n_heads
        self.norm = norm  # if True performs weight normalizatin instead of weight standardization
        self.partial_conv = partial_conv and kwargs.get("padding", 0) == 1  # implement only for pad=1 for now
        if self.partial_conv:
            assert self.weight.shape[2:] == (3, 3), "Partial conv only supports 3x3 conv"
            self.register_buffer("partial_mask", torch.zeros(1, 1, 1, 1))
        if coord_conv:
            self.register_buffer("coords", torch.zeros(1, 1, 1, 1))
        self.coord_conv = coord_conv

    def forward(self, x):

        if self.coord_conv:
            # need to check both spatial and batch dimension here
            if (self.coords.size(-1) != x.size(-1)) or (self.coords.size(0) != x.size(0)):
                self.coords = self._get_coords(x)
            x = torch.cat([x, self.coords], dim=1)

        if self.norm:
            # upd. on 01.09.21
            # maybe it didn't work because of centralization? maybe dividing by norm itself should be enough?
            weight = pt.utils.misc.zero_mean_conv_weight(self.weight)
            weight = weight / (weight.view(weight.size(0), -1).norm(dim=-1) + self.eps)[:, None, None, None]
            weight = weight * self.gain * self.scale
        else:
            weight = F.batch_norm(
                self.weight.view(1, self.out_channels, -1),
                None,
                None,
                weight=(self.gain * self.scale).view(-1),
                # weight=self.gain.view(-1), # gain is multiplied by scale during init
                training=True,
                momentum=0.0,
                eps=self.eps,
            ).reshape_as(self.weight)
        # skip adding bias for partial conv
        # proj = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # return proj

        bias = None if self.partial_conv else self.bias
        proj = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        if self.n_heads != 1:
            # this idea doesn't work maybe delete
            proj = proj.view(proj.size(0), self.n_heads, proj.size(1) // self.n_heads, proj.size(2), proj.size(3))
            proj = proj.mean(1)

        if self.partial_conv:
            if (proj.size(-1) != self.partial_mask.size(-1)):
                self.partial_mask = self._get_partial_mask(proj)
            # add bias only after masking
            proj = proj.mul(self.partial_mask)
            if self.bias is not None:
                proj = proj + self.bias.view(1, proj.size(1), 1, 1)
        return proj

    @torch.no_grad()
    def _get_partial_mask(self, inp: torch.Tensor):
        # Idea from `Partial Convolution based Padding` paper has too many extra details long-story short
        # we compensate for zero padding by slightly increasing output of convolution at the edges. that's it.
        mask = torch.ones(1, 1, inp.size(2), inp.size(3)).to(inp)
        kernel = torch.ones(1, 1, 3, 3).to(inp)
        mask = F.conv2d(mask, kernel, padding=1)
        return 9 / mask

    @torch.no_grad()
    def _get_coords(self, x):
        """Idea from coord conv. give model understanding of current absolute location on the image
        SOLO paper stated that single CoordConv is enough and gains from multiple of them are neglegible
        """
        xx = torch.linspace(-1, 1, x.size(-1)).expand(x.size(0), 1, x.size(2), x.size(3)).to(x)
        yy = torch.linspace(-1, 1, x.size(-2)).expand(x.size(0), 1, x.size(3), x.size(2)).to(x)
        yy = yy.transpose(-1, -2)
        return torch.cat([xx, yy], dim=1)

    def extra_repr(self):
        return super().extra_repr() + f", gamma={self.gamma}"


def scaled_conv3x3(in_chs, out_chs, padding=1, **extra_kwargs):
    """3x3 convolution with padding"""
    bias = extra_kwargs.pop("bias", True)
    return ScaledStdConv2d(in_chs, out_chs, kernel_size=3, padding=padding, bias=bias, **extra_kwargs)


def scaled_conv1x1(in_chs, out_chs, **extra_kwargs):
    """3x3 convolution with padding"""
    return ScaledStdConv2d(in_chs, out_chs, kernel_size=1, padding=0, bias=True, **extra_kwargs)


def wrapped_conv1x1(in_chs, out_chs, gain_init=None, gamma=None, **extra_kwargs):
    conv = conv1x1(in_chs, out_chs, **extra_kwargs)
    if gamma is None:
        return conv
    else:
        return nn.Sequential(OrderedDict([("conv", conv), ("gain", Gain(out_chs))]))


def wrapped_conv3x3(in_chs, out_chs, gain_init=None, gamma=None, **extra_kwargs):
    conv = conv3x3(in_chs, out_chs, **extra_kwargs)
    if gamma is None:
        return conv
    else:
        return nn.Sequential(OrderedDict([("conv", conv), ("gain", Gain(out_chs))]))


REMOVE_WN = True
REMOVE_WN = False
if REMOVE_WN:
    scaled_conv1x1 = wrapped_conv1x1
    scaled_conv3x3 = wrapped_conv3x3


class ChannelShuffle(nn.Module):
    """shuffles groups inside tensor. used to mix channels after grouped convolution. this is cheaper than using conv1x1
    Ref: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    """

    def __init__(self, groups=1):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        BS, CHS, H, W = x.shape
        return x.view(BS, self.groups, CHS // self.groups, H, W).transpose(1, 2).reshape_as(x)

    def extra_repr(self):
        return f"groups={self.groups}"


class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5, trainable=True):
        super().__init__()
        if trainable:
            self.register_parameter("scale", nn.Parameter(torch.ones(1)))
        else:
            self.scale = 1

        self.eps = eps

    def forward(self, x):
        norm = self.scale / x.norm(dim=1, keepdim=True).clamp(min=self.eps)
        return x * norm


class Affine(nn.Module):
    def __init__(self, value: float, trainable: bool = False):
        super().__init__()
        self.trainable = trainable
        if trainable:
            self.register_parameter("value", torch.nn.Parameter(torch.tensor(value)))
        else:
            self.register_buffer("value", torch.tensor(value))

    def forward(self, x):
        return x * self.value

    def extra_repr(self) -> str:
        return f"value={self.value}, trainable={self.trainable}"


class Gain(nn.Module):
    def __init__(self, size: float):
        super().__init__()
        self.register_parameter("gain", torch.nn.Parameter(torch.ones(1, size, 1, 1)))
        self.size = size

    def forward(self, x):
        return x * self.gain

    def extra_repr(self) -> str:
        return f"{self.size}"


class VarEMA(nn.Module):
    """Normalize tensor var by EMA of running vars"""

    # in first experiments it was 0.999 but it blew up
    def __init__(self, n_channels, use=True, decay=0.95, per_channel=False, eps=1e-4):
        super().__init__()
        self.decay = decay
        self.use = use
        self.per_channel = per_channel
        self.eps = eps
        self.register_buffer("std_ema", torch.ones(1, n_channels, 1, 1))
        self.register_buffer("mean_ema", torch.zeros(1, n_channels, 1, 1))
        if use:
            self.bn = nn.BatchNorm2d(n_channels, affine=False)

    def forward(self, x):
        with torch.no_grad():
            # channel wise statistics
            if self.training: # only record training stats
                std, mean = torch.std_mean(x, dim=(0, 2, 3), keepdim=True)
                self.std_ema = self.decay * self.std_ema + (1 - self.decay) * std
                self.mean_ema = self.decay * self.mean_ema + (1 - self.decay) * mean
        if self.use:
            if self.training:
                std = x.std(dim=(0, 2, 3), keepdim=True) # .mean().cpu().item()
                # like in Batch ReNormalization. this doesn't actually help with problems during backward
                # need manual backward as in MABN
                with torch.no_grad(): 
                    r = (std / self.std_ema).clamp(1/5, 5)
                return x / (std + self.eps) * r
            else:
                return x / self.std_ema
        else:  # needed to monitor running variance during training without any changes to the network
            return x


class MeanEMA(nn.Module):
    """Center tensor by EMA of running means per channel"""

    def __init__(self, decay=0.99):
        super().__init__()
        self.decay = decay

    def forward(self, x):
        # normalize per sample
        return x - x.mean(dim=(1, 2, 3), keepdim=True)
        # with torch.no_grad():
        #     mean = x.mean(dim=(0, 2, 3), keepdim=True)
        #     # maybe initialize
        #     if not hasattr(self, 'mean_ema'):
        #         self.register_buffer("mean_ema", torch.ones_like(mean))
        #     self.mean_ema = self.decay * self.mean_ema + (1 - self.decay) * mean
        # return x - self.mean_ema


class EMABlock(nn.Module):
    """Use simple EMA normalization before each block to remove variance shift"""

    def __init__(
        self,
        in_chs,
        out_chs,
        groups=1,
        groups_width=None,
        activation="relu",
        conv_kwargs=None,
        keep_prob=1,
        remove_ema=False,
        conv_act=False,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        groups = in_chs // groups_width if groups_width else groups
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        conv_kwargs["groups"] = groups
        self.varema = nn.Identity() if remove_ema else VarEMA()
        shuffle = ChannelShuffle(groups) if groups != 1 else nn.Identity()
        if conv_act:
            layers = [
                ("conv1", scaled_conv3x3(in_chs, out_chs, **conv_kwargs)),
                ("shuffle", shuffle),
                ("act1", activation_from_name(activation, inplace=False)),
                ("drop_path", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()),
            ]
        else:
            layers = [
                ("act1", activation_from_name(activation, inplace=False)),
                ("conv1", scaled_conv3x3(in_chs, out_chs, **conv_kwargs)),
                ("shuffle", shuffle),
                ("drop_path", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()),
            ]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        res = self.varema(x)
        out = self.block(res)
        if self.in_chs == self.out_chs:
            out += res
        else:
            out[:, : self.in_chs] += res
        return out


class XCA_mod(nn.Module):
    """Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    This could be viewed as dynamic 1x1 convolution
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = scaled_conv1x1(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = scaled_conv1x1(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # maybe want to have residual outside the class
        self.residual = residual

    def forward(self, x):
        B, C, H, W = x.shape
        # C` == channels per head, Hd == num heads
        # B x C x H x W -> B x 3*C x H x W -> B x 3 x Hd x C` x H*W -> 3 x B x Hd x C` x H*W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, -1).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # -> B x Hd x C` x C`
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # B x Hd x C` x C` @ B x Hd x C` x H*W -> B x C x H x W
        x_out = (attn @ v).reshape(B, C, H, W)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        # in original paper there is no residual here
        return x + x_out if self.residual else x_out

    def load_state_dict(self, state_dict):
        # to allow loading from Linear layer
        new_sd = {}
        for k, v in state_dict.items():
            if v.dim() == 2:
                new_sd[k] = v[..., None, None]
            else:
                new_sd[k] = v
        super().load_state_dict(new_sd)


class VGGBlock(nn.Module):
    """act - norm - conv (no residual). Would this even work?"""

    def __init__(
        self,
        in_chs,
        out_chs,
        groups_width=None,
        activation="relu",
        # like in Partial Residual defines which part of the output would receive residual
        # only 0.5 is supported for now
        conv_kwargs=None,
        # attn_kwargs=None,
        pre_norm=None,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        groups = max(in_chs // groups_width, 1) if groups_width else 1
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        conv_kwargs["groups"] = groups
        # maybe actually could use inplace here, it shouldn't matter
        self.block = nn.Sequential(
            pre_norm if pre_norm else nn.Identity(),
            activation_from_name(activation, inplace=False),
            scaled_conv3x3(in_chs, out_chs, **conv_kwargs),
            ChannelShuffle(groups) if groups != 1 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class ConvMixBlock(nn.Module):
    """Here we go again. Another day another module. Would be close to ConvActBlock but with other 
    parameters, don't want to clutter and create one large class"""

    def __init__(
        self,
        in_chs,
        out_chs,
        groups_width=None,
        activation="relu",
        # like in Partial Residual defines which part of the output would receive residual
        # only 0.5 is supported for now
        partial_factor=1.0,
        conv_kwargs=None,
        # attn_kwargs=None,
        pre_norm=None,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.n_common_chs = min(in_chs, out_chs)
        groups = max(in_chs // groups_width, 1) if groups_width else 1
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        conv_kwargs["groups"] = groups
        self.pre_norm = pre_norm if pre_norm else nn.Identity()
        self.conv = scaled_conv3x3(in_chs, out_chs, **conv_kwargs)
        self.shuffle = ChannelShuffle(groups) if groups != 1 else nn.Identity()
        # maybe actually could use inplace here, it shouldn't matter
        self.act = activation_from_name(activation, inplace=False)
        assert partial_factor in {0, 0.5, 1}, "only {0, 0.5, 1} is supported as partial factor"
        self.partial_factor = partial_factor

    def forward(self, x):
        out = self.act(x)
        out = self.pre_norm(out)
        out = self.shuffle(self.conv(out))
        
        if self.partial_factor == 1:
            out[:, :self.n_common_chs] = out[:, :self.n_common_chs] + x[:, :self.n_common_chs]
        elif self.partial_factor == 0:
            # no residual
            out = out
        elif self.partial_factor == 0.5:
            res_chs = int(self.n_common_chs * 0.5)
            out[:, :res_chs] = out[:, :self.res_chs] + x[:, :self.res_chs]
        # out = self.attn(out)
        return out


class ConvActBlock(nn.Module):
    """conv + residual -> Act. allows fusing convolution with residual later"""

    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        groups=1,
        groups_width=None,
        activation="relu",
        conv_kwargs=None,
        attn_kwargs=None,
        pre_norm=None,
    ):  # , keep_prob=1):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        groups = max(in_chs // groups_width, 1) if groups_width else groups
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        conv_kwargs["groups"] = groups
        self.pre_norm = pre_norm if pre_norm else None
        self.res_downscale = BlurPool(in_chs) if stride == 2 else nn.Identity()
        self.conv = scaled_conv3x3(in_chs, out_chs, stride=stride, **conv_kwargs)
        self.shuffle = ChannelShuffle(groups) if groups != 1 else nn.Identity()
        # maybe actually could use inplace here, it shouldn't matter
        self.act = activation_from_name(activation, inplace=False)
        self.attn = XCA_mod(dim=out_chs, **attn_kwargs) if attn_kwargs is not None else nn.Identity()

    def forward(self, x):
        x_block = x
        if self.pre_norm:
            x_block = self.pre_norm(x_block)
        out = self.shuffle(self.conv(x_block))
        res = self.res_downscale(x)
        if self.in_chs == self.out_chs:
            out += res
        else:
            out[:, : self.in_chs] += res
        out = self.act(out)
        out = self.attn(out)
        return out


# instead of using skipinit gain or alpha we can simply init last conv gain with +-alpha to simplify
class NormFreeBlock(nn.Module):
    """BasicBlock with preactivatoin & without downsample support"""

    def __init__(
        self,
        in_chs,
        out_chs,
        mid_chs=None,
        groups=1,
        groups_width=None,
        activation="relu",
        attention_type=None,
        attention_kwargs=None,  # pass something else to attention
        attention_gain=2.0,
        keep_prob=1,
        beta=1.0,
        alpha=0.2,
        conv_kwargs=None,
        pre_norm_group_width=None,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        mid_chs = mid_chs or out_chs
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        groups = in_chs // groups_width if groups_width else groups
        attn_kw = attention_kwargs if attention_kwargs is not None else {}
        attn_layer = (
            nn.Sequential(get_attn(attention_type)(mid_chs, **attn_kw), Affine(attention_gain))
            if attention_type
            else nn.Identity()
        )
        layers = [
            # don't want to modify residual, so not in-place
            ("act1", activation_from_name(activation, inplace=False)),
            ("conv1", scaled_conv3x3(in_chs, mid_chs, gain_init=beta, groups=groups, **conv_kwargs)),
            ("shuffle1", ChannelShuffle(groups) if groups > 1 else nn.Identity()),
            ("act2", activation_from_name(activation)),
            ("conv2", scaled_conv3x3(mid_chs, out_chs, gain_init=alpha, groups=groups, **conv_kwargs)),
            ("shuffle2", ChannelShuffle(groups) if groups > 1 else nn.Identity()),
            ("attn", attn_layer),
            ("drop_path", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()),
        ]
        self.block = nn.Sequential(OrderedDict(layers))
        if pre_norm_group_width is None:
            self.pre_norm = nn.Identity()
        else:
            pre_norm_groups = in_chs // pre_norm_group_width
            self.pre_norm = nn.GroupNorm(num_groups=pre_norm_groups, num_channels=in_chs)

    def forward(self, x):
        out = self.block(self.pre_norm(x))
        if self.in_chs == self.out_chs:
            out += x
        else:
            out[:, : self.in_chs] += x
        return out


class NormFreeBlockTimm(nn.Module):
    """Close to default block of Timm but without downsampling support"""

    def __init__(
        self,
        in_chs,
        out_chs,
        mid_chs=None,
        groups=1,
        groups_width=None,
        activation="relu",
        attention_type=None,
        attention_kwargs=None,  # pass something else to attention
        attention_gain=2.0,
        keep_prob=1,
        conv_kwargs=None,
        beta=1.0,
        alpha=0.2,
        regnet_attention=False,  # RegNet puts attention in bottleneck
        pre_norm_group_width=None,
        full_conv=False,  # if True set padding=2 in first conv and padding=0 in second one
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        mid_chs = mid_chs or out_chs
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        groups = mid_chs // groups_width if groups_width else groups
        attn_kw = attention_kwargs if attention_kwargs is not None else {}
        attn_layer = (
            nn.Sequential(get_attn(attention_type)(mid_chs, **attn_kw), Affine(attention_gain))
            if attention_type
            else nn.Identity()
        )
        # idea is from "On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location"
        # where they have shown than Full-Conv is better than same-conv. But their idea leads to huge increase in size of feature map
        # so i'm applying full-conv + valid conv instead of same-conv + same-conv
        # upd. changing meaning of `full_conv` flag to use reflect convolution
        pad1, pad2 = (1, 1)  # (2, 0) if full_conv else (1, 1)
        if full_conv:
            conv_kwargs["padding_mode"] = "reflect"
        layers = [
            # don't want to modify residual, so not in-place
            ("act1", activation_from_name(activation, inplace=False)),
            ("conv1", scaled_conv1x1(in_chs, mid_chs, gain_init=beta, **conv_kwargs)),
            ("act2", activation_from_name(activation)),
            ("conv2", scaled_conv3x3(mid_chs, mid_chs, groups=groups, padding=pad1, **conv_kwargs)),
            ("act2b", activation_from_name(activation)),
            ("conv2b", scaled_conv3x3(mid_chs, mid_chs, groups=groups, padding=pad2, **conv_kwargs)),
            ("attn1", attn_layer if regnet_attention else nn.Identity()),
            ("act3", activation_from_name(activation)),
            ("conv3", scaled_conv1x1(mid_chs, out_chs, gain_init=alpha, **conv_kwargs)),
            ("attn2", attn_layer if not regnet_attention else nn.Identity()),
            ("drop_path", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()),
        ]
        self.block = nn.Sequential(OrderedDict(layers))
        if pre_norm_group_width is None:
            self.pre_norm = nn.Identity()
        else:
            pre_norm_groups = in_chs // pre_norm_group_width
            self.pre_norm = nn.GroupNorm(num_groups=pre_norm_groups, num_channels=in_chs)

    def forward(self, x):
        out = self.block(self.pre_norm(x))
        if self.in_chs == self.out_chs:
            out += x
        else:
            out[:, : self.in_chs] += x
        return out


class PreInvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        mid_chs=None,
        keep_prob=1,
        norm_layer=ABN,
        norm_act="relu",  # drop connect param
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
            ("drop_path", DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()),
        ]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        out = self.block(x)
        if self.in_chs == self.out_chs:
            out += x
        else:
            out[:, : self.in_chs] += x
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


def update_dict(to_dict: Dict, from_dict: Dict) -> Dict:
    """close to `to_dict.update(from_dict)` but correctly updates internal dicts"""
    for k, v in from_dict.items():
        if hasattr(v, "keys") and k in to_dict.keys():
            to_dict[k].update(v)
        else:
            to_dict[k] = v
    return to_dict


def test_update_dict():
    """Tests to make sure updating dict works as expected"""
    # simple update
    d_to = {"a": 10, "b": 20}
    d_from = {"a": 12, "c": 30}
    d_expected = {"a": 12, "b": 20, "c": 30}
    assert update_dict(d_to, d_from) == d_expected

    # recursive update. dict.update would fail in this case
    d_to = {"foo": {"a": 10, "b": 20}}
    d_from = {"foo": {"a": 12, "c": 30}}
    d_expected = {"foo": {"a": 12, "b": 20, "c": 30}}
    assert update_dict(d_to, d_from) == d_expected

    # when key is not present in `to`
    d_to = {"bar": 1}
    d_from = {"foo": {"a": 12, "c": 30}}
    d_expected = {"bar": 1, "foo": {"a": 12, "c": 30}}
    assert update_dict(d_to, d_from) == d_expected


class CModel(nn.Module):
    """
    Args:
        features_idx (List[int]):
            from which layers to additionally save features
    """

    def __init__(
        self,
        layer_config: List[LayerDef],
        extra_kwargs: Dict[str, Dict] = None,
        features_idx: List[int] = None,
    ):
        super().__init__()
        if not isinstance(layer_config[0], LayerDef):
            layer_config = [LayerDef(*l) for l in layer_config]
        if extra_kwargs is not None:
            self._update_config_with_extra_params(layer_config, extra_kwargs)
        layers, saved = self._parse_config(layer_config)
        self.layers: nn.Module = layers
        self.saved: List[int] = saved
        self._patch_drop_path()

    def _patch_drop_path(self, drop_path_name="drop_path"):
        """DropPath works best when it linearly increased each block. Expects than drop_path layer is already created (by passing keep_prob"""
        keep_probs = [m.keep_prob for n, m in self.layers.named_modules() if drop_path_name in n]
        if len(keep_probs) == 0:  # skip if no drop_path
            return
        max_keep_prob = max(keep_probs)
        num_drop_layers = len(keep_probs)
        keep_prob_rates = torch.linspace(max_keep_prob, 1, num_drop_layers).numpy().tolist()
        for n, m in self.layers.named_modules():
            if drop_path_name in n:
                m.keep_prob = keep_prob_rates.pop()

    @staticmethod
    def _update_config_with_extra_params(layer_config: List[LayerDef], extra_kwargs: Dict[str, Dict]):
        for l_name, l_kwargs in extra_kwargs.items():
            for l in layer_config:
                if l.module == l_name:
                    # kwargs from layer should overwrite global extra_kwargs
                    l.kwargs = update_dict(deepcopy(l_kwargs), l.kwargs)

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
# layer_config = [
#     (-1, 1, "pt.modules.SpaceToDepth", 2),  # 0
#     (-1, 1, "conv3x3", (12, 64)),  # 0
#     (-1, 1, "pt.modules.BlurPool", 64),
#     (-1, 6, "pt.modules.residual.FusedRepVGGBlock", (64, 64)),
#     (-1, 1, "pt.modules.BlurPool", 64),
#     (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (64, 128)),
#     (-1, 7, "pt.modules.residual.FusedRepVGGBlock", (128, 128)),
#     (-1, 1, "pt.modules.BlurPool", 128),
#     (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (128, 256)),
#     (-1, 11, "pt.modules.residual.FusedRepVGGBlock", (256, 256)),
#     (-1, 1, "pt.modules.BlurPool", 256),
#     (-1, 1, "pt.modules.residual.FusedRepVGGBlock", (256, 512)),
#     (-1, 5, "pt.modules.residual.FusedRepVGGBlock", (512, 512)),
#     (-1, 1, "pt.modules.FastGlobalAvgPool2d", (), dict(flatten=True)),
#     (-1, 1, "nn.Dropout", (), dict(p=0, inplace=False)),
#     (-1, 1, "nn.Linear", (512, 1000)),
# ]

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


# model = CModel(layer_config).cuda()
# # print(model)
# inp = torch.rand(16, 3, 224, 224).cuda()

# model(inp).shape
