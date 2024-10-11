"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import  DropBlock2d, DropPath, AvgPool2dSame, create_attn, create_classifier, trunc_normal_
from .registry import register_model

## for DWT 
import torch_dwt as tdwt

__all__ = ['ResNet_DWT_SE', 'BasicBlock30', 'Bottleneck30', 'SEModule']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet50': _cfg(
        # https://download.pytorch.org/models/resnet50-19c8e357.pth (RESNET)
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        #url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        interpolation='bicubic', crop_pct=0.95),
}
    
class SEModule(nn.Module):
    def __init__(self, channels=64, reduction_=1, loss_option=0):
        super(SEModule, self).__init__()
        
        self.eps = 1e-9
        reduction = int(reduction_ // 10)
        mean_true = int(reduction_ % 10)

        print(f'Reduction: {reduction} GT_OPTION: {mean_true}')

        if mean_true == 0:
            self.gt_lst = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # ALL Median
        elif mean_true == 1: 
            self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # All Highest <==== We choose this option
        elif mean_true == 2:
            self.gt_lst = [self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps] # All Lowest
        elif mean_true == 3:
            self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, self.eps, 1.0, 0.5, 0.5, self.eps, 1.0, self.eps, self.eps, self.eps] # LL is More Important
        elif mean_true == 4:
            self.gt_lst = [self.eps, self.eps, self.eps, self.eps, self.eps, 0.5, 0.5, 1.0, self.eps, 0.5, 0.5, 1.0, self.eps, 1.0, 1.0, 1.0] # HH is More Important
            

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, (channels * 2), kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, frequency_index, gt_lst=None):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x) # Excitation
        x = self.relu(x)
        x = self.fc2(x) # Scaling 
        x = self.sigmoid(x)
        
        mean = x[:, ::2]  # Even Position of output channels (Mean)
        std = x[:, 1::2]  # Odd Position of output channels (Standard Deviation)

        if self.training:
            g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean()
            return (module_input * mean), [g_loss, mean]

        else: # Validation 
            return (module_input * mean), (mean, std)
        
    
    def _gaussian_dist_pdf(self, data_point, var, freq_idx, gt_lst=None):
        if gt_lst == None:
            gt = self.gt_lst[freq_idx]
        elif isinstance(gt_lst, list):
            gt = gt_lst[freq_idx]
        else:
            gt = gt_lst
        var = var.clone() + self.eps
        pdf_value = torch.exp(- (data_point - gt) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
        return torch.clamp(pdf_value, min=self.eps)

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

class BasicBlock30(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_affine=True, 
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock30, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        
        self.bn1 = norm_layer(first_planes, affine=norm_affine)

        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        
        self.bn2 = norm_layer(outplanes, affine=norm_affine)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x, seq=0):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
           shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck30(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_affine=True,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, se_dwt_level=[0,0,0,0]):  # se_dwt_level Args =>  [self.dwt_level[0], self.dwt_bn[0], self.dwt_bn[1]]
        super(Bottleneck30, self).__init__()

        self.dwt_level = se_dwt_level[0]
        self.dwt_bn = se_dwt_level[1]
        self.reduction_option = se_dwt_level[2]

        if self.dwt_level == 0:
            self.split_count = 0
        elif self.dwt_level == 1:
            self.split_count = 4
        elif self.dwt_level == 2:
            self.split_count = 16
        self.nll_loss = None
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        
        self.bn1 = norm_layer(first_planes, affine=norm_affine)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
                first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
                padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)

        self.bn2 = norm_layer(width, affine=norm_affine)
        
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes, affine=norm_affine)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        
        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., se_dwt_level=[0,0,0,0], **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            if stage_idx == (se_dwt_level[3]-1) and block_idx == 0:
                # print(f'stage_dix {stage_idx} blocks idx: {block_idx}')
                blocks.append(block_fn(
                    inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0. else None, se_dwt_level=se_dwt_level, **block_kwargs))
            else:
                blocks.append(block_fn(
                    inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info
    

# Learnable parameter를 위하여 설계됨.
class ResNet_DWT_SE(nn.Module):
    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1, no_skip=False, aux_header=False,
            dwt_kernel_size=[0,0,0], dwt_level=[2,2,2], dwt_bn=[0,0,0], deep_format=False, 
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None, mvar=False, meta_option=0, **kwargs):
        super(ResNet_DWT_SE, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.in_channel = in_chans
        
        SE_NET = SEModule

        # DWT Options #
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        self.dwt_bn = dwt_bn
        
        self.nll_loss = None
    
        dwt_conv_layer = list()
        dwt_ada_layer = list()
        print(f'Resnet_DWT initialize -> Kenrel: {self.dwt_kernel_size} / Level: {self.dwt_level} / BN: {self.dwt_bn}' )

        # Matching for Intermediated features size 
        if self.dwt_kernel_size[0] == 7:
            self.conv1_padding = 3
        elif self.dwt_kernel_size[0] == 5:
            self.conv1_padding = 2
        elif self.dwt_kernel_size[0] == 3:
            self.conv1_padding = 1
        else:
            self.conv1_padding = 3
        
        # Matching for Intermediated features size and channels
        if self.dwt_level[0] == 1:
            self.split_count = 4
            self.split_im_size = 56
        elif self.dwt_level[0] == 2:
            self.split_count = 16
            self.split_im_size = 28
        elif self.dwt_level[0] == 3:
            self.split_count = 64
            self.split_im_size = 14
        elif self.dwt_level[0] == 0:
            self.split_count = 0
        else:
            assert "DWT_ LEVEL Assertion not (0, 1, 2)"

        if self.dwt_bn[0] == 2: 
            self.is_normal_attention = True # GCAL Only Routine
        else:
            self.is_normal_attention = False # without GCAL and DEL
            
        inplanes = 64

        if self.dwt_kernel_size[0] == 0:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            if self.is_normal_attention:
                self.normal_attention = SE_NET(reduction_=int(81))
        else:
            if self.dwt_bn[0] >= 3:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=1, padding=self.conv1_padding, bias=False)
                self.se1 = SE_NET(reduction_=self.dwt_bn[1])
            else:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False)
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            if self.dwt_level[0] == 1:
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False).cuda())
                    dwt_ada_layer.append(SE_NET(reduction_=self.dwt_bn[1]))
                    if self.dwt_bn[0] == 1:
                        break
            elif self.dwt_level[0] == 2:
                stride_n = 2
                if self.dwt_bn[0] >= 3:
                    stride_n = 1
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=stride_n, padding=self.conv1_padding, bias=False).cuda())
                    dwt_ada_layer.append(SE_NET(reduction_=self.dwt_bn[1]))
                    if self.dwt_bn[0] == 1:
                        break

            elif self.dwt_level[0] == 3: 
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False).cuda())
            else:     
                assert 'DWT LEVEL Aseertion'
            self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)
            self.dwt_ada_layer = nn.ModuleList(dwt_ada_layer)

        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        if self.dwt_bn[2] == 0:
            self.se_dwt_level = [0,0,0,0]
        else:
            self.se_dwt_level = [self.dwt_level[0], self.dwt_bn[0], self.dwt_bn[1], self.dwt_bn[2]] #LEVEL? FIXED? REDUCTION? POSITION?
        
        header_in_channel = 512
        channels = [64, 128, 256, header_in_channel]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, se_dwt_level = self.se_dwt_level, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = header_in_channel * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)        

        self.init_weights(zero_init_last=zero_init_last, dwt_kernel_size=self.dwt_kernel_size[0])

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True, dwt_kernel_size=0):
        if dwt_kernel_size == 1:
            trunc_normal_(self.pos_embed, std=.02)

        if dwt_kernel_size != 0 and dwt_kernel_size != 1:
            if len(self.dwt_conv_layer) != 0:
                for idx, item in enumerate(self.dwt_conv_layer):
                    nn.init.kaiming_normal_(item.weight, mode='fan_out', nonlinearity='relu')
            
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
            
    def get_first_output_fm(self):
        return self.first_output_fm

    def get_nll_loss(self):
        return self.nll_loss

    def DEL(self, if_map):
        split_tensor_lst = list()

        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst)
        
        output_tensor_lst = [self.dwt_conv_layer[0](split_tensor_lst[i]) for i in range(self.split_count)]
        
        for i in range(self.split_count):
            output_tensor_lst[i], nll_loss_ = self.dwt_ada_layer[0](output_tensor_lst[i], i)

            if self.training == True:
                nll_loss = nll_loss_[0]
                if i == 0:
                    self.nll_loss = nll_loss
                else:
                    self.nll_loss += nll_loss
            else:
                if i == 0:
                    self.nll_loss = list()
                    self.nll_loss.append(nll_loss_)
                else:
                    self.nll_loss.append(nll_loss_)

        if self.dwt_level[0] == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level[0] == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
        elif self.dwt_level[0] == 3:
            return tdwt.get_dwt_level3_inverse(output_tensor_lst)

    def disable_grad_sequential(self, sequential_module):
        for module in sequential_module:
            for param in module.parameters():
                param.requires_grad = False
    
    def disable_grad_item(self, module):
            for param in module.parameters():
                param.requires_grad = False
        
    def forward_features(self, x):
        if self.dwt_kernel_size[0] == 0:
            x = self.conv1(x)
            if self.is_normal_attention: # GCAL Processing w/o DEL
                x, nll_loss_ = self.normal_attention(x, 0)

                if self.training == True:
                    self.nll_loss = nll_loss_[0]
                else:
                    self.nll_loss = list()
                    self.nll_loss.append(nll_loss_)         
        else:
            x = self.DEL(x) # GCAL Processing with DEL
        
        with torch.no_grad():
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
            else:
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if isinstance(x, list):
            par_x = list()
            for item in x:
                x = self.global_pool(item)
                if self.drop_rate:
                    x = F.dropout(x, p=float(self.drop_rate), training=self.training)
                par_x.append(self.fc(x))
            return par_x
        else:
            x = self.global_pool(x)
            if self.drop_rate:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            return x if pre_logits else self.fc(x)

    def forward(self, x, is_test=False):
        if is_test:
            self.disable_grad_item(self.bn1)
            self.disable_grad_item(self.act1)
            self.disable_grad_item(self.maxpool)
            self.disable_grad_sequential(self.layer1)
            self.disable_grad_sequential(self.layer2)
            self.disable_grad_sequential(self.layer3)
            self.disable_grad_sequential(self.layer4)
            self.disable_grad_item(self.fc)
            x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        return self.forward_head(x)

def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet_DWT_SE, variant, pretrained, **kwargs)

@register_model
def resnet50_dwt_se(pretrained=False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, meta_option=0, **kwargs):
    """Constructs a ResNet-50 model. [3, 4, 6, 3]
    """
    model_args = dict(block=Bottleneck30, layers=[3, 4, 6, 3], aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, meta_option=0, **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)

@register_model
def resnet26_dwt_se(pretrained=False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, meta_option=0, **kwargs):
    """Constructs a ResNet-26 model. [2, 2, 2, 2]
    """
    model_args = dict(block=Bottleneck30, layers=[2, 2, 2, 2], aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, meta_option=meta_option, **kwargs)
    return _create_resnet('resnet26', pretrained, **model_args)