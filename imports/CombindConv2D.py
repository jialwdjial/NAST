import os
from turtle import forward
import torch
from torch import nn, tensor
import torch.nn.functional
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

class BayarConv2D(nn.Module):
    def __init__(self ,inputchanel, outputchanel, kernelsize) :
        super(BayarConv2D,self).__init__()


        self.mask = None
        weight = torch.Tensor(inputchanel, outputchanel, kernelsize, kernelsize)
        self.weight = torch.nn.Parameter(weight)
        nn.init.xavier_normal_(self.weight)
        # print(self.weight)
    
    def _initialize_mask(self) :
        chanelin = self.weight.shape[0]
        chanelout  = self.weight.shape[1]
        ksize = self.weight.shape[2] 
        m = np.zeros([chanelin, chanelout, ksize, ksize]).astype('float32')
        m[:,:,ksize//2,ksize//2] = 1.
        self.mask = torch.tensor(m).cuda()
    
    def _get_new_weight(self) :
        with torch.no_grad():
            if self.mask is None :
                self._initialize_mask()
            self.weight.data *= (1-self.mask)
            # print(self.weight)
            rest_sum = torch.sum(self.weight, dim=(2,3), keepdims=True)
            # print('sum')
            # print(rest_sum)
            # print(rest_sum.shape)
            self.weight.data /= rest_sum + 1e-7
            self.weight.data -= self.mask
            # print(self.weight)
            # print(self.weight.grad)
    
    def forward(self, x):
        self._get_new_weight()
        return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)


class SRMConv2D(nn.Module):
    def _get_srm_list(self) :
        # srm kernel 1                                                                                                                                
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2                                                                                                                                
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3                                                                                                                                
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]
    
    def _build_SRM_kernel(self) :
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate( srm_list ):
            for ch in range(3) :
                this_ch_kernel = np.zeros([5,5,3]).astype('float32')
                this_ch_kernel[:,:,ch] = srm
                kernel.append( this_ch_kernel )
        kernel = np.stack( kernel, axis=-1 )

        kernel = np.swapaxes(kernel,1,2)

        kernel = np.swapaxes(kernel,0,3)      
        return kernel
    
    def __init__(self):
        super(SRMConv2D,self).__init__()
        self.weight = torch.tensor(self._build_SRM_kernel()).cuda()
    def forward(self, x):
        with torch.no_grad():
            return torch.nn.functional.conv2d(x, weight=self.weight, padding = 2)

class CombindConv2D(nn.Module):
    def __init__(self, outputChanels) -> None:
        super(CombindConv2D, self).__init__()
        self.subLayer1 = BayarConv2D(3,3,5) # outchanel 3
        self.relu1 = nn.ReLU(inplace=True)
        self.subLayer2 = SRMConv2D()        # outchanel 9 
        self.relu2 = nn.ReLU(inplace=True)
        self.subLayer3 = nn.Conv2d(3,outputChanels - 3 - 9, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.subLayer1(x)
        x1 = self.relu1(x1)
        x2 = self.subLayer2(x)
        x2 = self.relu2(x2)
        x3 = self.subLayer3(x)
        x3 = self.relu3(x3)

        x = torch.cat([x1,x2,x3], dim=1)
        # print(x.shape)
        return x
