""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward
import torch_dwt as tdwt

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingCrossEntropyWithDWT(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyWithDWT, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor, ratio: torch.Tensor, ratio_target) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
               
        #print('DWT \\ ')                      
        pred_lst = tdwt.get_dwt_level2_list(ratio)

        loss_func = nn.MSELoss()
        for i in range(16):
            ratio_target[i] = ratio_target[i].unsqueeze(-1)
            ratio_target[i] = ratio_target[i].to(torch.float32)
            if i == 0:  
                loss_dwt = loss_func(pred_lst[i], ratio_target[i].cuda())
            else:
                loss_dwt += loss_func(pred_lst[i], ratio_target[i].cuda())
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss# + loss_dwt
        #print('-- ',loss.shape)
        #print('dwt-- ',loss_dwt.shape)
        return loss.mean(), loss_dwt
    
    
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
