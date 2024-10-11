"""VGG

Adapted from https://github.com/pytorch/vision 'vgg.py' (BSD-3-Clause) with a few changes for
timm functionality.

Copyright 2021 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, checkpoint_seq
from .fx_features import register_notrace_module
from .layers import ClassifierHead
from .registry import register_model

import torch_dwt as tdwt
import numpy as np
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 
    'vgg19_dwt', 'SEModule', # JIN Added
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'vgg11': _cfg(url='https://download.pytorch.org/models/vgg11-bbd30ac9.pth'),
    'vgg13': _cfg(url='https://download.pytorch.org/models/vgg13-c768596a.pth'),
    'vgg16': _cfg(url='https://download.pytorch.org/models/vgg16-397923af.pth'),
    'vgg19': _cfg(url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
    'vgg11_bn': _cfg(url='https://download.pytorch.org/models/vgg11_bn-6002323d.pth'),
    'vgg13_bn': _cfg(url='https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'),
    'vgg16_bn': _cfg(url='https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'),
    'vgg19_bn': _cfg(url='https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'),

    'vgg19_dwt': _cfg(url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
}


cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19_dwt': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class SEModule(nn.Module):

    def __init__(self, channels=64, reduction_=1, loss_option=0):
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

            return (module_input * mean), [g_loss, mean]

        else: # Validation 
            target_mean = self.gt_lst[frequency_index]
            batch_size = mean.size(0)  
            target_tensor = torch.full((batch_size, 64, 1), target_mean).squeeze(-1).cuda().detach()
            return (module_input * mean), mean
        
    
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

@register_notrace_module  # reason: FX can't symbolically trace control flow in forward method
class ConvMlp(nn.Module):

    def __init__(
            self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0,
            drop_rate: float = 0.2, act_layer: nn.Module = None, conv_layer: nn.Module = None):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1]))
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class VGG(nn.Module):

    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            mlp_ratio: float = 1.0,
            act_layer: nn.Module = nn.ReLU,
            conv_layer: nn.Module = nn.Conv2d,
            norm_layer: nn.Module = None,
            global_pool: str = 'avg',
            drop_rate: float = 0., 
            no_skip=False, aux_header=False, 
            dwt_kernel_size=[0,0,0], dwt_level=[2,2,2], dwt_bn=[0,0,0], deep_format=False, 
            mvar=False, meta_option=0,
    ) -> None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []
        stem_layers: List[nn.Module] = []

        self.nll_loss = None
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        self.dwt_bn = dwt_bn

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

        # if self.dwt_kernel_size[0] != 0:
        #     self.se_ada_module= SE
        if self.dwt_kernel_size[0] > 0:
            self.dwt_ada_layer = SEModule(reduction_=self.dwt_bn[1])
        for idx, v in enumerate(cfg):
            last_idx = len(layers) - 1
            if v == 'M':
                #print(f'JIN LOG[M] -> {idx} / {v}')
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                #print(f'JIN LOG -> {idx} / {v}')
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    if idx == 0 or idx == 1:
                        stem_layers += [conv2d, norm_layer(v), act_layer(inplace=True)]    
                        layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                    else:
                        layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                else:
                    if idx == 0 or idx == 1:
                        stem_layers += [conv2d, act_layer(inplace=True)]
                        layers += [conv2d, act_layer(inplace=True)]
                    else:
                        layers += [conv2d, act_layer(inplace=True)]
                prev_chs = v
        self.stem_features = nn.Sequential(*stem_layers)
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))

        self.pre_logits = ConvMlp(
            prev_chs, self.num_features, 7, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, act_layer=act_layer, conv_layer=conv_layer)
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        self._initialize_weights()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        # this treats BN layers as separate groups for bn variants, a lot of effort to fix that
        return dict(stem=r'^features\.0', blocks=r'^features\.(\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
    
    def get_nll_loss(self):
        return self.nll_loss

    def dwt_rearrange(self, if_map):
        split_tensor_lst = list()

        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst)

        output_tensor_lst = [self.stem_features[0](split_tensor_lst[i]) for i in range(len(split_tensor_lst))]
        output_tensor_lst2 = [self.stem_features[1](output_tensor_lst[i]) for i in range(len(split_tensor_lst))]

        del output_tensor_lst
        for i in range(len(split_tensor_lst)):
            output_tensor_lst2[i], nll_loss_ = self.dwt_ada_layer(output_tensor_lst2[i],i)

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
                return tdwt.get_dwt_level1_inverse(output_tensor_lst2)
        elif self.dwt_level[0] == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst2)
        elif self.dwt_level[0] == 3:
            return tdwt.get_dwt_level3_inverse(output_tensor_lst2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.dwt_kernel_size[0] == 0:
            #x = self.stem_features(x)
            x = self.features(x)
        else:
            x = self.dwt_rearrange(x)
            for idx, layer in enumerate(self.features):
                if idx >= 2:  # 첫 두 레이어를 건너뛰고 나머지 레이어 적용
                    x = layer(x)
        
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _filter_fn(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        k_r = k
        k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
        k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
        k_r = k_r.replace('classifier.6', 'head.fc')
        if 'classifier.0.weight' in k:
            v = v.reshape(-1, 512, 7, 7)
        if 'classifier.3.weight' in k:
            v = v.reshape(-1, 4096, 1, 1)
        out_dict[k_r] = v
    return out_dict


def _create_vgg(variant: str, pretrained: bool, **kwargs: Any) -> VGG:
    cfg = variant.split('_')[0]
    # NOTE: VGG is one of few models with stride==1 features w/ 6 out_indices [0..5]
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3, 4, 5))
    model = build_model_with_cfg(
        VGG, variant, pretrained,
        model_cfg=cfgs[cfg],
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        pretrained_filter_fn=_filter_fn,
        **kwargs)
    return model


@register_model
def vgg11(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg11', pretrained=pretrained, **model_args)


@register_model
def vgg11_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm2d, **kwargs)
    return _create_vgg('vgg11_bn', pretrained=pretrained, **model_args)


@register_model
def vgg13(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg13', pretrained=pretrained, **model_args)


@register_model
def vgg13_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm2d, **kwargs)
    return _create_vgg('vgg13_bn', pretrained=pretrained, **model_args)


@register_model
def vgg16(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg16', pretrained=pretrained, **model_args)


@register_model
def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm2d, **kwargs)
    return _create_vgg('vgg16_bn', pretrained=pretrained, **model_args)


@register_model
def vgg19(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg19', pretrained=pretrained, **model_args)

@register_model
def vgg19_dwt(pretrained: bool = False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, meta_option=0, **kwargs: Any) -> VGG:
    
    model_args = dict(aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, meta_option=0,**kwargs)
    return _create_vgg('vgg19', pretrained=pretrained, **model_args)


@register_model
def vgg19_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(norm_layer=nn.BatchNorm2d, **kwargs)
    return _create_vgg('vgg19_bn', pretrained=pretrained, **model_args)