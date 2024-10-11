""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
import torch
from .helpers import to_2tuple
from .trace_utils import _assert
import torch_dwt as tdwt
import numpy as np
class SEModule(nn.Module):

    def __init__(self, channels=128, reduction_=1, loss_option=0):
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


    def forward(self, x, frequency_index, gt_lst=None):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x) # Excitation
        x = self.relu(x)
        x = self.fc2(x) # scaling 
        # 임시테스트 
        x = self.sigmoid(x)
        
        mean = x[:, ::2]  # 모든 배치의 짝수 위치 (평균)
        std = x[:, 1::2]  # 모든 배치의 홀수 위치 (표준편차)

        self.mean_analysis = mean
        self.std_analysis = std

        if self.training:
            g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean()
            return (module_input * mean), [g_loss, mean]

        else: # Validation
            return (module_input * mean), (mean, std)#mean#[mean, std]
        
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
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            dwt_level=[0, 0, 0],
            dwt_kernel_size=[0, 0, 0],        
            no_skip=False, aux_header=False, dwt_bn=[0,0,0], deep_format=False
    ):
        super().__init__()
        
        SE_NET = SEModule
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        self.dwt_bn = dwt_bn
        self.nll_loss = None
        print(f'LOG_CHECKER = SWIN3 - ? {dwt_kernel_size} / {dwt_level} / {dwt_bn}')
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        if self.dwt_level[0] == 0:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
            self.se_attn = SE_NET(reduction_=self.dwt_bn[1])
                
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

                                              
    def dwt_rearrange(self, if_map):
        split_tensor_lst = list()

        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst)
        
        output_tensor_lst = [self.proj(split_tensor_lst[i]) for i in range(len(split_tensor_lst))]
        means_lst = list()
        for i in range(len(split_tensor_lst)):
            # if self.is_mean != None:
            #     output_tensor_lst[i], nll_loss_ = self.se_attn(output_tensor_lst[i], i, self.is_mean)
            # else:
            output_tensor_lst[i], nll_loss_ = self.se_attn(output_tensor_lst[i], i)

            if self.training == True:
                nll_loss = nll_loss_[0]
                means_lst.append(nll_loss_[1])
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
                                              
    def get_nll_loss(self):
        return self.nll_loss

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #print('Before projection data shape --> ', x.shape)
        
        if self.dwt_level[0] == 0:
            x = self.proj(x)
        else:
            x = self.dwt_rearrange(x)
                                              
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
                                                  
    # def init_weights(self):
    #     if self.dwt_kernel_size[0] != 0:
    #         for idx, item in enumerate(self.dwt_conv_layer):
    #             nn.init.kaiming_normal_(item.weight, mode='fan_out', nonlinearity='relu')