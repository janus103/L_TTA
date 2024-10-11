""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
import functools
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer

import numpy as np
import torch_dwt as tdwt
import torch

class SEModule(nn.Module):

    def __init__(self, channels=16, reduction_=1, loss_option=0):
        super(SEModule, self).__init__()
        
        self.eps = 1e-9
        self.loss_option = loss_option
        reduction = int(reduction_ // 10)
        mean_true = int(reduction_ % 10)

        self.mean_analysis = None
        self.std_analysis = None

        print(f'Reduction: {reduction} GT_OPTION: {mean_true}')

        if mean_true == 0:
            self.gt_lst = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # ALL Median
        elif mean_true == 1: 
            self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # All Highest
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
        

    def adaptive_std(self, std):
        threshold = 0.9
        valid_std_mask = std >= threshold
        
        std = (std * valid_std_mask) + self.eps
        
        gt_lst = torch.ones((128, 64, 1, 1)).cuda()
        gt_lst = (gt_lst * valid_std_mask) + self.eps

        return std, gt_lst


    def forward(self, x, frequency_index, gt_lst=None,is_mean=None):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x) # Excitation
        x = self.relu(x)
        x = self.fc2(x) # scaling 
        x = self.sigmoid(x)

        mean = x[:, ::2]  # 모든 배치의 짝수 위치 (평균)
        std = x[:, 1::2]  # 모든 배치의 홀수 위치 (표준편차)

        self.mean_analysis = mean
        self.std_analysis = std

        if self.training:
            if is_mean != None:
                target_mean = is_mean[frequency_index]
            else:
                target_mean = self.gt_lst[frequency_index]
            batch_size = mean.size(0)  
            target_tensor = torch.full((batch_size, 64, 1), target_mean).squeeze(-1).cuda().detach()
            mean_mean = mean.mean(dim=(2,3))
            if self.loss_option % 10 == 0:
                g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean()
            else:
                g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean() + F.mse_loss(mean_mean,target_tensor)
            #target_tensor = target_tensor.unsqueeze(-1).unsqueeze(-1)
            #stand_value = (mean - target_tensor) / (std)
            #return (module_input * stand_value), [g_loss, mean]
            return (module_input * mean), [g_loss, mean]

        else: # Validation 
            target_mean = self.gt_lst[frequency_index]
            batch_size = mean.size(0)  
            target_tensor = torch.full((batch_size, 64, 1), target_mean).squeeze(-1).cuda().detach()
            # mean_mean = mean.mean(dim=(2,3))
            # target_tensor = target_tensor.unsqueeze(-1).unsqueeze(-1)
            # stand_value = (mean - target_tensor) / (std)
            return (module_input * mean), mean#[mean, std]
        
    
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

class ConvNormAct(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_layer=None, dwt_level=0):
        super(ConvNormAct, self).__init__()
        self.dwt_level = dwt_level
        print(f'CLASS_ {ConvNormAct}: DWT_LEVEL => {self.dwt_level}')
        self.nll_los = None
        if dwt_level == 2 or dwt_level == 3:
            print('MobileViT => DWT MODE with level 2')
            self.dwt_ada_layer = SEModule(reduction_=41)
            self.dwt_kernel_size = 3

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def get_nll_loss(self):
        return self.nll_loss

    def dwt_rearrange(self, if_map):
        split_tensor_lst = list()

        if self.dwt_level == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)
        elif self.dwt_level == 3:
            split_tensor_lst.append(if_map)

        output_tensor_lst = [self.conv(split_tensor_lst[i]) for i in range(len(split_tensor_lst))]

        for i in range(len(split_tensor_lst)):
            output_tensor_lst[i], nll_loss_ = self.dwt_ada_layer(output_tensor_lst[i],i)

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
        
        if self.dwt_level == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
        elif self.dwt_level == 3:
            return output_tensor_lst[0]

    def forward(self, x):
        if self.dwt_level == 0:
            x = self.conv(x)
        else:
            x = self.dwt_rearrange(x)
        x = self.bn(x)
        return x


ConvBnAct = ConvNormAct


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if isinstance(aa_layer, functools.partial):
        if issubclass(aa_layer.func, nn.AvgPool2d):
            return aa_layer()
        else:
            return aa_layer(channels)
    elif issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class ConvNormActAa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, aa_layer=None, drop_layer=None):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None and stride == 2

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)
        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=use_aa)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.aa(x)
        return x
