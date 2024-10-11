def dwt_rearrange_dyn(self, if_map, dwt_ratio, dwt_quant=1, dwt_drop=False, w=None):
        split_tensor_lst = list()

        LL, hs = self.DWT(if_map)
        B, N, C, H, W = hs.shape
        hs = hs.reshape(B, N*C, H, W)
        LH, HL, HH= torch.split(hs, self.in_channel, dim=1)

        LL_LL, LL_HS = self.DWT(LL)
        B, N, C, H, W = LL_HS.shape
        LL_HS = LL_HS.reshape(B, N*C, H, W)
        LL_LH, LL_HL, LL_HH = torch.split(LL_HS, self.in_channel, dim=1)
        split_tensor_lst.append(LL_LL)
        split_tensor_lst.append(LL_LH)
        split_tensor_lst.append(LL_HL)
        split_tensor_lst.append(LL_HH)

        LH_LL, LH_HS = self.DWT(LH)
        B, N, C, H, W = LH_HS.shape
        LH_HS = LH_HS.reshape(B, N*C, H, W)
        LH_LH, LH_HL, LH_HH = torch.split(LH_HS, self.in_channel, dim=1)
        split_tensor_lst.append(LH_LL)
        split_tensor_lst.append(LH_LH)
        split_tensor_lst.append(LH_HL)
        split_tensor_lst.append(LH_HH)

        HL_LL, HL_HS = self.DWT(HL)
        B, N, C, H, W = HL_HS.shape
        HL_HS = HL_HS.reshape(B, N*C, H, W)
        HL_LH, HL_HL, HL_HH = torch.split(HL_HS, self.in_channel, dim=1)
        split_tensor_lst.append(HL_LL)
        split_tensor_lst.append(HL_LH)
        split_tensor_lst.append(HL_HL)
        split_tensor_lst.append(HL_HH)

        HH_LL, HH_HS = self.DWT(HH)
        B, N, C, H, W = HH_HS.shape
        HH_HS = HH_HS.reshape(B, N*C, H, W)
        HH_LH, HH_HL, HH_HH = torch.split(HH_HS, self.in_channel, dim=1)
        split_tensor_lst.append(HH_LL)
        split_tensor_lst.append(HH_LH)
        split_tensor_lst.append(HH_HL)
        split_tensor_lst.append(HH_HH)

        if self.dwt_bn[0] == 0: 
            output_tensor_lst = [self.dwt_conv_layer[i](split_tensor_lst[i]) for i in range(self.split_count)]
            for i in range(self.split_count):
                
                output_tensor_lst[i], nll_loss = self.dwt_ada_layer[0](output_tensor_lst[i],i)
                if self.training == True: 
                    if i == 0:
                        self.nll_loss = nll_loss
                    else:
                        self.nll_loss += nll_loss
                else:
                    if i == 0:
                        self.nll_loss = []
                        self.nll_loss.append(nll_loss)
                    else:
                        self.nll_loss.append(nll_loss)
        else:
            output_tensor_lst = [self.dwt_conv_layer[0](split_tensor_lst[i]) for i in range(self.split_count)]
            for i in range(self.split_count):
                output_tensor_lst[i], nll_loss = self.dwt_ada_layer[0](output_tensor_lst[i],i)
                if self.training == True:
                    if i == 0:
                        self.nll_loss = nll_loss
                    else:
                        self.nll_loss += nll_loss
                else:
                    if i == 0:
                        self.nll_loss = nll_loss
                    else:
                        self.nll_loss.append(nll_loss)

        LL_HS = torch.stack([output_tensor_lst[1], output_tensor_lst[2], output_tensor_lst[3]], dim=2)
        LH_HS = torch.stack([output_tensor_lst[5], output_tensor_lst[6], output_tensor_lst[7]], dim=2)
        HL_HS = torch.stack([output_tensor_lst[9], output_tensor_lst[10], output_tensor_lst[11]], dim=2)
        HH_HS = torch.stack([output_tensor_lst[13], output_tensor_lst[14], output_tensor_lst[15]], dim=2)

        LL = self.DWT(output_tensor_lst[0], LL_HS)
        LH = self.DWT(output_tensor_lst[4], LH_HS)
        HL = self.DWT(output_tensor_lst[8], HL_HS)
        HH = self.DWT(output_tensor_lst[12], HH_HS)

        HS = torch.stack([LH, HL, HH], dim=2)
        out = self.DWT(LL, HS)

        return out