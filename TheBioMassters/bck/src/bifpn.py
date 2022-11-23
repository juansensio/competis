import torch 
import torch.nn as nn
from timm.models.layers import create_conv2d, create_pool2d, get_act_layer
from typing import List, Callable, Optional, Union, Tuple
import torch.nn.functional as F
import logging
from functools import partial
import itertools
from omegaconf import OmegaConf

_ACT_LAYER = get_act_layer('silu')

class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        # super(ConvBnAct2d, self).__init__()
        super().__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.
    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)
    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        # super(Interpolate2d, self).__init__()
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False)

# upsample or downsample an input feature map based on reduction_ratio

# Here is the general idea:
# if out_channels is not equal to in_channels, then use a 1x1 convolution operation to make them the same. 
# if the reduction ratio is not equal to 1, then either upsample or downsample the input feature map. 
# If reduction_ratio<1 then, Upsample the input, otherwise if reduction_ratio>1 then, Downsample the input.

class ResampleFeatureMap(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, reduction_ratio=1., pad_type='', downsample=None, upsample=None,
            norm_layer=nn.BatchNorm2d, apply_bn=False, conv_after_downsample=False, redundant_bias=False):
        # super(ResampleFeatureMap, self).__init__()
        super().__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias, act_layer=None)

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            if downsample in ('max', 'avg'):
                stride_size = int(reduction_ratio)
                downsample = create_pool2d(
                     downsample, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type)
            else:
                downsample = Interpolate2d(scale_factor=1./reduction_ratio, mode=downsample)
            self.add_module('downsample', downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))

class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        # super(Fnode, self).__init__()
        super().__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        z = self.combine(x)
        y = self.after_combine(z)
        return y

class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 conv_after_downsample=False, redundant_bias=False, weight_method='attn'):
        # super(FpnCombine, self).__init__()
        super().__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample, redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)
        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out

class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        # super(SeparableConv2d, self).__init__()
        super().__init__()
        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return 
        
class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER,
                 apply_resample_bn=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True, redundant_bias=False):
        # super(BiFpnLayer, self).__init__()
        super().__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False
        self.feature_info = []
        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            reduction = fnode_cfg['reduction']
            combine = FpnCombine(
                feature_info, fpn_config, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction, pad_type=pad_type, downsample=downsample, upsample=upsample,
                norm_layer=norm_layer, apply_resample_bn=apply_resample_bn, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=False, norm_layer=norm_layer, act_layer=act_layer)
            if not conv_bn_relu_pattern:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            # after_combine.add_module(
            #     'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))
            after_combine.add_module(
                'conv', ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]

class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        # super(SequentialList, self).__init__(*args)
        super().__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x
        
def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))
    return p

def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.
    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.
    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = [level_last_id(i), level_last_id(i + 1)] if i != max_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = [level_last_id(i), level_last_id(i - 1)] if i != min_level else [level_last_id(i)]
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    return 

def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.
    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).
    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))

    i = max_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [
                level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [node_ids[i][2], node_ids[i][4]],
            'weight_method': quad_method
        })
        node_ids[i].append(next(id_cnt))

    return p

def get_fpn_config(fpn_name, min_level=3, max_level=7):
    p = OmegaConf.create()
    # p.nodes = [
    #     {'reduction': 64, 'inputs_offsets': [3, 4], 'weight_method': 'fastattn'}, 
    #     {'reduction': 32, 'inputs_offsets': [2, 5], 'weight_method': 'fastattn'}, 
    #     {'reduction': 16, 'inputs_offsets': [1, 6], 'weight_method': 'fastattn'}, 
    #     {'reduction': 8, 'inputs_offsets': [0, 7], 'weight_method': 'fastattn'}, 
    #     {'reduction': 16, 'inputs_offsets': [1, 7, 8], 'weight_method': 'fastattn'}, 
    #     {'reduction': 32, 'inputs_offsets': [2, 6, 9], 'weight_method': 'fastattn'}, 
    #     {'reduction': 64, 'inputs_offsets': [3, 5, 10], 'weight_method': 'fastattn'}, 
    #     {'reduction': 128, 'inputs_offsets': [4, 11], 'weight_method': 'fastattn'}
    # ]
    p.nodes = [
        {'reduction': 16, 'inputs_offsets': [3, 4], 'weight_method': 'fastattn'}, 
        {'reduction': 8, 'inputs_offsets': [2, 5], 'weight_method': 'fastattn'}, 
        {'reduction': 4, 'inputs_offsets': [1, 6], 'weight_method': 'fastattn'}, 
        {'reduction': 2, 'inputs_offsets': [0, 7], 'weight_method': 'fastattn'}, 
        {'reduction': 4, 'inputs_offsets': [1, 7, 8], 'weight_method': 'fastattn'}, 
        {'reduction': 8, 'inputs_offsets': [2, 6, 9], 'weight_method': 'fastattn'}, 
        {'reduction': 16, 'inputs_offsets': [3, 5, 10], 'weight_method': 'fastattn'}, 
        {'reduction': 32, 'inputs_offsets': [4, 11], 'weight_method': 'fastattn'}
    ]
    return p
    # if not fpn_name:
    #     fpn_name = 'bifpn_fa'
    # name_to_config = {
    #     'bifpn_sum': bifpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
    #     'bifpn_attn': bifpn_config(min_level=min_level, max_level=max_level, weight_method='attn'),
    #     'bifpn_fa': bifpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    #     'pan_sum': panfpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
    #     'pan_fa': panfpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    #     'qufpn_sum': qufpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
    #     'qufpn_fa': qufpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    # }
    # return name_to_config[fpn_name]

class BiFpn(nn.Module):

    # def __init__(self, config, feature_info):
    def __init__(self, feature_info):
        # super(BiFpn, self).__init__()
        super().__init__()
        self.num_levels = 5 #config.num_levels
        self.fpn_channels = 64
        self.fpn_cell_repeats = 3
        # norm_layer = config.norm_layer or nn.BatchNorm2d
        norm_layer = nn.BatchNorm2d
        # if config.norm_kwargs:
        #     norm_layer = partial(norm_layer, **config.norm_kwargs)
        # act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        act_layer = _ACT_LAYER
        # fpn_config = config.fpn_config or get_fpn_config(
        #     config.fpn_name, min_level=config.min_level, max_level=config.max_level)
        fpn_config = get_fpn_config(None)

        self.resample = nn.ModuleDict()
        for level in range(self.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=self.fpn_channels,
                    # pad_type=config.pad_type,
                    # downsample=config.downsample_type,
                    # upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    # apply_bn=config.apply_resample_bn,
                    # conv_after_downsample=config.conv_after_downsample,
                    # redundant_bias=config.redundant_bias,
                )
                in_chs = self.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        for rep in range(self.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=self.fpn_channels,
                # num_levels=config.num_levels,
                # pad_type=config.pad_type,
                # downsample=config.downsample_type,
                # upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                # separable_conv=config.separable_conv,
                # apply_resample_bn=config.apply_resample_bn,
                # conv_after_downsample=config.conv_after_downsample,
                # conv_bn_relu_pattern=config.conv_bn_relu_pattern,
                # redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x