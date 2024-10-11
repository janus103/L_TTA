import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward, DWTInverse
import torchvision.transforms as T

def make_figure_out(tensor_dwt_lst):
    tensor_dwt_lst[0] = tensor_dwt_lst[0].squeeze(0)
    tensor_dwt_lst[1] = tensor_dwt_lst[1].squeeze(0)
    tensor_dwt_lst[2] = tensor_dwt_lst[2].squeeze(0)
    tensor_dwt_lst[3] = tensor_dwt_lst[3].squeeze(0)
    
    # tensor_dwt_lst[0] = (torch.round(tensor_dwt_lst[0] / 2).to(torch.int32))
    tensor_dwt_lst[0] = (torch.round(tensor_dwt_lst[0] / 2))
    # tensor_dwt_lst[1] = (torch.round(tensor_dwt_lst[1] + 127.5).to(torch.int32))
    tensor_dwt_lst[1] = (torch.round(tensor_dwt_lst[1] + 127.5))
    # tensor_dwt_lst[2] = (torch.round(tensor_dwt_lst[2] + 127.5).to(torch.int32))
    tensor_dwt_lst[2] = (torch.round(tensor_dwt_lst[2] + 127.5))
    # tensor_dwt_lst[3] = (torch.round(tensor_dwt_lst[3] + 127.5).to(torch.int32))
    tensor_dwt_lst[3] = (torch.round(tensor_dwt_lst[3] + 127.5))

def get_dwt_pil(image_pil):
    image_tensor = T.functional.pil_to_tensor(image_pil)
    tensor_dwt_lst = list()

    get_dwt_level1(image_tensor.unsqueeze(0).cuda().to(torch.float32), tensor_dwt_lst ,None, None)

    print('XXX')
    return torch.cat(tensor_dwt_lst, dim=0)



def get_dwt_level1(x: torch.Tensor, lst, x_dwt_rate = None, x_bias: torch.Tensor = None):
    
    ratio = x.float()
    
    xfm = DWTForward(J=1, mode='zero', wave='db1').cuda()
    LL, HS = xfm(ratio)
    if x_dwt_rate == None:
        #rate = [0.5, 0.25, 0.125]
        #rate = [0.5, 0., 0.]
        rate = [1, 1, 1]
        
        #print('Level 1 ')
        lst.append(LL)
        lst.append(HS[0][:,:,0,:,:] * rate[0])
        lst.append(HS[0][:,:,1,:,:] * rate[0])
        lst.append(HS[0][:,:,2,:,:] * rate[1])
    elif x_dwt_rate == True:      
        rate = [0.75, 0.75, 0.75]
        
        #print('Level 1 ')
        lst.append(LL)
        lst.append(HS[0][:,:,0,:,:] * rate[0])
        lst.append(HS[0][:,:,1,:,:] * rate[1])
        lst.append(HS[0][:,:,2,:,:] * rate[2])
    else:
        rate = [1, 1, 1]
        
        #print('Level 1 ')
        lst.append(LL)
        lst.append(HS[0][:,:,0,:,:] * rate[0])
        lst.append(HS[0][:,:,1,:,:] * rate[1])
        lst.append(HS[0][:,:,2,:,:] * rate[2])
        #lst.append(LL)
        # lst.append(HS[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
        # lst.append(HS[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
        # lst.append(HS[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))
def get_dwt_level1_CPU(x: torch.Tensor, lst, x_dwt_rate = None, x_bias: torch.Tensor = None):
    
    ratio = x.float()
    
    xfm = DWTForward(J=1, mode='zero', wave='db1')#.cuda()
    LL, HS = xfm(ratio)
    
    if x_dwt_rate == None:
        #rate = [0.5, 0.25, 0.125]
        #rate = [0.5, 0., 0.]
        rate = [1, 1, 1]
        
        #print('Level 1 ')
        lst.append(LL)
        lst.append(HS[0][:,:,0,:,:] * rate[0])
        lst.append(HS[0][:,:,1,:,:] * rate[0])
        lst.append(HS[0][:,:,2,:,:] * rate[1])
    else:      
        lst.append(LL)
        lst.append(HS[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
        lst.append(HS[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
        lst.append(HS[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))
    #make_figure_out(lst)
def get_dwt_level3(x: torch.Tensor, lst, x_dwt_rate = None, x_bias: torch.Tensor = None):
    
    temp_lst = list()
    ratio = x
    get_dwt_level1(ratio, temp_lst, x_dwt_rate, x_bias) # LL LH HL HH
    
    
    level2_lst = list()
    for i in range(4):      
        temp_lst_1 = []
        get_dwt_level1(temp_lst[i], temp_lst_1, x_dwt_rate, x_bias)
        level2_lst.append(temp_lst_1[0])
        level2_lst.append(temp_lst_1[1])
        level2_lst.append(temp_lst_1[2])
        level2_lst.append(temp_lst_1[3])
    
    for i in range(16):
        temp_lst_1 = []
        get_dwt_level1(level2_lst[i], temp_lst_1, x_dwt_rate, x_bias)
        lst.append(temp_lst_1[0])
        lst.append(temp_lst_1[1])
        lst.append(temp_lst_1[2])
        lst.append(temp_lst_1[3])

    
def get_dwt_level3_inverse(lst):
    lev_2_lst = []
    for i in range(16):
        temp_lst = []
        for j in range(4):
            idx = i*4+j
            temp_lst.append(lst[idx])
        lev_2_lst.append(get_dwt_level1_inverse(temp_lst))
    
    lev_1_lst = []
    for i in range(4):
        temp_lst = []
        for j in range(4):
            idx = i*4+j
            temp_lst.append(lev_2_lst[idx])
        lev_1_lst.append(get_dwt_level1_inverse(temp_lst))
    
    return get_dwt_level1_inverse(lev_1_lst)
    
def get_dwt_level1_inverse(lst):
    ifm = DWTInverse(mode='zero', wave='db1').cuda()
    #B, C, W, H = lst[0].shape
    lst[1] = lst[1].unsqueeze(2) 
    lst[2] = lst[2].unsqueeze(2) 
    lst[3] = lst[3].unsqueeze(2) 
    final_H = torch.cat([lst[1],lst[2],lst[3]], dim=2)
    
    final = ifm((lst[0],[final_H]))    
    return final

def get_dwt_level2(x: torch.Tensor, lst, x_dwt_rate = None, x_quant = None):
    
    ratio = x
    
    xfm = DWTForward(J=1, mode='zero', wave='db1').cuda()
    LL, HS = xfm(ratio)

    HS_0 = HS[0][:,:,0,:,:]
    HS_1 = HS[0][:,:,1,:,:]
    HS_2 = HS[0][:,:,2,:,:]

    LL1, HS1 = xfm(LL)
    LL2, HS2 = xfm(HS_0)
    LL3, HS3 = xfm(HS_1)
    LL4, HS4 = xfm(HS_2)
    
    #print('?? HS4 length ', len(HS4)) # length = 1 
    #if x_dwt_rate == None:
        # if x_quant == None:
        #     rate = [0.5, 0.25, 0.125]
        # elif x_quant == 0:
        #     rate = [1, 1, 1]
        # elif x_quant == 1: 
        #     rate = [0.5, 0.25, 0.125]
        # elif x_quant == 2: 
        #     rate = [0.5, 0, 0]
        # elif x_quant == 3: 
        #     rate = [0, 0, 0]
        # else:
        #     rate = [1, 1, 1]
    rate = [1, 1, 1]
    lst.append(LL1)
    lst.append(HS1[0][:,:,0,:,:] * rate[0])
    lst.append(HS1[0][:,:,1,:,:] * rate[0])
    lst.append(HS1[0][:,:,2,:,:] * rate[2])

    #HS2
    lst.append(LL2)
    lst.append(HS2[0][:,:,0,:,:] * rate[1])
    lst.append(HS2[0][:,:,1,:,:] * rate[1])
    lst.append(HS2[0][:,:,2,:,:] * rate[2])

    #HS3
    lst.append(LL3)
    lst.append(HS3[0][:,:,0,:,:] * rate[1])
    lst.append(HS3[0][:,:,1,:,:] * rate[1])
    lst.append(HS3[0][:,:,2,:,:] * rate[2])

    #HS4
    lst.append(LL4)
    lst.append(HS4[0][:,:,0,:,:] * rate[1])
    lst.append(HS4[0][:,:,1,:,:] * rate[1])
    lst.append(HS4[0][:,:,2,:,:] * rate[2])
    # elif x_dwt_rate == True:
    #     rate = [0,0,0]
        
    #     lst.append(LL1)
    #     lst.append(HS1[0][:,:,0,:,:] * rate[0])
    #     lst.append(HS1[0][:,:,1,:,:] * rate[0])
    #     lst.append(HS1[0][:,:,2,:,:] * rate[2])

    #     #HS2
    #     lst.append(LL2)
    #     lst.append(HS2[0][:,:,0,:,:] * rate[1])
    #     lst.append(HS2[0][:,:,1,:,:] * rate[1])
    #     lst.append(HS2[0][:,:,2,:,:] * rate[2])

    #     #HS3
    #     lst.append(LL3)
    #     lst.append(HS3[0][:,:,0,:,:] * rate[1])
    #     lst.append(HS3[0][:,:,1,:,:] * rate[1])
    #     lst.append(HS3[0][:,:,2,:,:] * rate[2])

    #     #HS4
    #     lst.append(LL4)
    #     lst.append(HS4[0][:,:,0,:,:] * rate[1])
    #     lst.append(HS4[0][:,:,1,:,:] * rate[1])
    #     lst.append(HS4[0][:,:,2,:,:] * rate[2])
    # else:      
    #     #print('dwt_rate == Not None')
    #     lst.append(LL1)
    #     lst.append(HS1[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
    #     lst.append(HS1[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
    #     lst.append(HS1[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))

    #     #HS2
    #     lst.append(LL2)
    #     lst.append(HS2[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
    #     lst.append(HS2[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
    #     lst.append(HS2[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))

    #     #HS3
    #     lst.append(LL3)
    #     lst.append(HS3[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
    #     lst.append(HS3[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
    #     lst.append(HS3[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))

    #     #HS4
    #     lst.append(LL4)
    #     lst.append(HS4[0][:,:,0,:,:] * (sum(x_dwt_rate[1]) / len(x_dwt_rate[1])))
    #     lst.append(HS4[0][:,:,1,:,:] * (sum(x_dwt_rate[2]) / len(x_dwt_rate[2])))
    #     lst.append(HS4[0][:,:,2,:,:] * (sum(x_dwt_rate[3]) / len(x_dwt_rate[3])))
    
def get_dwt_level2_inverse(lst, mode=0):
    ifm = DWTInverse(mode='zero', wave='db1').cuda()
    B, C, W, H = lst[0].shape    
    lst[1] = lst[1].unsqueeze(2) 
    lst[2] = lst[2].unsqueeze(2) 
    lst[3] = lst[3].unsqueeze(2) 
    output1 = torch.cat([lst[1],lst[2],lst[3]], dim=2)
    
    lst[5] = lst[5].unsqueeze(2) 
    lst[6] = lst[6].unsqueeze(2) 
    lst[7] = lst[7].unsqueeze(2) 
    output2 = torch.cat([lst[5],lst[6],lst[7]], dim=2)
    
    
    lst[9] = lst[9].unsqueeze(2) 
    lst[10] = lst[10].unsqueeze(2) 
    lst[11] = lst[11].unsqueeze(2) 
    output3 = torch.cat([lst[9],lst[10],lst[11]], dim=2)
    
    lst[13] = lst[13].unsqueeze(2) 
    lst[14] = lst[14].unsqueeze(2)   
    lst[15] = lst[15].unsqueeze(2) 
    output4 = torch.cat([lst[13],lst[14],lst[15]], dim=2)
    if mode == 0:
        output5 = ifm((lst[0],[output1]))
        output6 = ifm((lst[4],[output2]))
        output7 = ifm((lst[8],[output3]))
        output8 = ifm((lst[12],[output4]))
    elif mode == 1:
        return ifm((lst[0],[output1]))
    
    elif mode == 2:
        lst[4] = lst[4].unsqueeze(2)
        lst[8] = lst[8].unsqueeze(2)
        lst[12] = lst[12].unsqueeze(2)
        LL_output = torch.cat([lst[4],lst[8],lst[12]], dim=2)
        return ifm((lst[0],[LL_output]))
        
    
    
    output6 = output6.unsqueeze(2) 
    output7 = output7.unsqueeze(2)   
    output8 = output8.unsqueeze(2) 
    final_H = torch.cat([output6, output7, output8], dim=2)
    
    final = ifm((output5,[final_H]))    
    return final
    
    
def get_dwt_level1_list(ratio_x: torch.Tensor):
    ratio = ratio_x
    B,C,H,W = ratio.shape

    size = W
    
    ratio = ratio.reshape((B,C,size,size))        

    xfm = DWTForward(J=1, mode='zero', wave='db1').cuda()
    LL, HS = xfm(ratio)

    LL_sum = LL.sum(-1).sum(-1)
    
    HS_0 = HS[0][:,:,0,:,:]
    HS_1 = HS[0][:,:,1,:,:]
    HS_2 = HS[0][:,:,2,:,:]

    HS_0_sum = HS_0.sum(-1).sum(-1)
    HS_1_sum = HS_1.sum(-1).sum(-1)
    HS_2_sum = HS_2.sum(-1).sum(-1)

    total_sum = HS_0_sum + HS_1_sum + HS_2_sum + LL_sum
    
    LL_ratio = LL_sum/total_sum
    HS_0_ratio = HS_0_sum/total_sum
    HS_1_ratio = HS_1_sum/total_sum
    HS_2_ratio = HS_2_sum/total_sum   

    pred_lst = [
        LL_ratio,\
        HS_0_ratio,\
        HS_1_ratio,\
        HS_2_ratio
    ]
    
    return pred_lst

def get_dwt_level2_list(ratio_x: torch.Tensor):
    ratio = ratio_x
    B,C,H,W = ratio.shape

    size = W
    
    ratio = ratio.reshape((B,C,size,size))        

    xfm = DWTForward(J=1, mode='zero', wave='db1').cuda()
    LL, HS = xfm(ratio)

    HS_0 = HS[0][:,:,0,:,:]
    HS_1 = HS[0][:,:,1,:,:]
    HS_2 = HS[0][:,:,2,:,:]

    LL1, HS1 = xfm(LL)
    LL2, HS2 = xfm(HS_0)
    LL3, HS3 = xfm(HS_1)
    LL4, HS4 = xfm(HS_2)

    HS1_0 = HS1[0][:,:,0,:,:]
    HS1_1 = HS1[0][:,:,1,:,:]
    HS1_2 = HS1[0][:,:,2,:,:]

    HS2_0 = HS2[0][:,:,0,:,:]
    HS2_1 = HS2[0][:,:,1,:,:]
    HS2_2 = HS2[0][:,:,2,:,:]

    HS3_0 = HS3[0][:,:,0,:,:]
    HS3_1 = HS3[0][:,:,1,:,:]
    HS3_2 = HS3[0][:,:,2,:,:]

    HS4_0 = HS4[0][:,:,0,:,:]
    HS4_1 = HS4[0][:,:,1,:,:]
    HS4_2 = HS4[0][:,:,2,:,:]

    HS1_0_sum = HS1_0.sum(-1).sum(-1)
    HS1_1_sum = HS1_1.sum(-1).sum(-1)
    HS1_2_sum = HS1_2.sum(-1).sum(-1)

    HS2_0_sum = HS2_0.sum(-1).sum(-1)
    HS2_1_sum = HS2_1.sum(-1).sum(-1)
    HS2_2_sum = HS2_2.sum(-1).sum(-1)

    HS3_0_sum = HS3_0.sum(-1).sum(-1)
    HS3_1_sum = HS3_1.sum(-1).sum(-1)
    HS3_2_sum = HS3_2.sum(-1).sum(-1)

    HS4_0_sum = HS4_0.sum(-1).sum(-1)
    HS4_1_sum = HS4_1.sum(-1).sum(-1)
    HS4_2_sum = HS4_2.sum(-1).sum(-1)

    LL1_sum = LL1.sum(-1).sum(-1)
    LL2_sum = LL2.sum(-1).sum(-1)
    LL3_sum = LL3.sum(-1).sum(-1)
    LL4_sum = LL4.sum(-1).sum(-1)

    total_sum = HS1_0_sum + HS1_1_sum + HS1_2_sum + HS2_0_sum + HS2_1_sum + HS2_2_sum + HS3_0_sum + HS3_1_sum + HS3_2_sum + HS4_0_sum + HS4_1_sum + HS4_2_sum + LL1_sum + LL2_sum + LL3_sum + LL4_sum
    HS1_0_ratio = HS1_0_sum/total_sum
    HS1_1_ratio = HS1_1_sum/total_sum
    HS1_2_ratio = HS1_2_sum/total_sum
    HS2_0_ratio = HS2_0_sum/total_sum
    HS2_1_ratio = HS2_1_sum/total_sum
    HS2_2_ratio = HS2_2_sum/total_sum
    HS3_0_ratio = HS3_0_sum/total_sum
    HS3_1_ratio = HS3_1_sum/total_sum
    HS3_2_ratio = HS3_2_sum/total_sum
    HS4_0_ratio = HS4_0_sum/total_sum
    HS4_1_ratio = HS4_1_sum/total_sum
    HS4_2_ratio = HS4_2_sum/total_sum
    LL1_ratio = LL1_sum/total_sum
    LL2_ratio = LL2_sum/total_sum
    LL3_ratio = LL3_sum/total_sum
    LL4_ratio = LL4_sum/total_sum   

    pred_lst = [
        LL1_ratio,\
        HS1_0_ratio,\
        HS1_1_ratio,\
        HS1_2_ratio,\
        LL2_ratio,\
        HS2_0_ratio,\
        HS2_1_ratio,\
        HS2_2_ratio,\
        LL3_ratio,\
        HS3_0_ratio,\
        HS3_1_ratio,\
        HS3_2_ratio,\
        LL4_ratio,\
        HS4_0_ratio,\
        HS4_1_ratio,\
        HS4_2_ratio
    ]
    
    return pred_lst