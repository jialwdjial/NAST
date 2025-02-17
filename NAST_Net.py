import os
from matplotlib.pyplot import imshow
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import conv2d, dropout, nn, sigmoid, tensor
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
# Layer files
from imports.ParametersManager import *  # import training-help tools
from imports.CombindConv2D import *  # import defination of special layers
from imports.ZPool2D import *  # inport Z-Pooling layers
from imports.convlstm import *  # Copied from https://github.com/ndrplz/ConvLSTM_pytorch
import torch.nn.functional as F
from test import MobileViTAttention
operation_canditates = {
    '00': lambda filter1, filter2, stride, dilation: SeparableConv2d(filter1, filter2, 3, stride, dilation),
    '01': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 3, stride, 1),
    '02': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 5, stride, 2),
    '03': lambda filter1, filter2, stride, dilation: Identity(),
}
# Hyperparameter
ZPoolingWindows = [7, 15, 31]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)


class ConvX(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
                 is_training):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides,
                              padding=padding, dilation=dilate_rate)
        self.bn_in = bn_in
        if (self.bn_in == 'bn'):
            self.bn_layer = nn.BatchNorm2d(num_features=filters, affine=True)
        if (self.bn_in == 'in'):
            self.in_layer = nn.InstanceNorm2d(num_features=filters, affine=True)
        self.act_layer = nn.ReLU(inplace=True)  # 不保存中间变量

    def forward(self, x):
        x = self.conv(x)
        if (self.bn_in == 'bn'):
            x = self.bn_layer(x)
        if (self.bn_in == 'in'):
            x = self.in_layer(x)
        x = self.act_layer(x)
        return x


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, genotype=None):
        super(Block, self).__init__()
        if not genotype:
            genotype = ['03', '03', '03']

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(operation_canditates[genotype[i]](filters, filters, 1, dilation))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        rep.append(self.relu)
        rep.append(operation_canditates[genotype[2]](filters, filters, stride, 1))
        rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class CustomizedConv(nn.Module):
    def __init__(self, channels=1, choice='similarity'):
        super(CustomizedConv, self).__init__()
        self.channels = channels
        self.choice = choice
        kernel = [[0.03598, 0.03735, 0.03997, 0.03713, 0.03579],
                  [0.03682, 0.03954, 0.04446, 0.03933, 0.03673],
                  [0.03864, 0.04242, 0.07146, 0.04239, 0.03859],
                  [0.03679, 0.03936, 0.04443, 0.03950, 0.03679],
                  [0.03590, 0.03720, 0.04003, 0.03738, 0.03601]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.kernel = nn.modules.utils._pair(3)
        self.stride = nn.modules.utils._pair(1)
        self.padding = nn.modules.utils._quadruple(0)
        self.same = False

    def __call__(self, x):
        if self.choice == 'median':
            x = F.pad(x, self._padding(x), mode='reflect')
            x = x.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1])
            x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        else:
            x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding


# L2Norm Layer
class L2Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        a = torch.norm(x, 2, keepdim=True)  # 对于整个通道层求L2 Norm，并利用其进行标准化
        x = x / a
        return x


class Global_Local_Attention(nn.Module):
    def __init__(self):
        super(Global_Local_Attention, self).__init__()
        self.local_att = CustomizedConv(256, choice='similarity')

    def forward(self, x):
        tmp = x
        former = tmp.permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        num_0 = torch.einsum('bik,bjk->bij', [former, former])
        norm_former = torch.einsum("bij,bij->bi", [former, former])
        den_0 = torch.sqrt(torch.einsum('bi,bj->bij', [norm_former, norm_former]))
        cosine = num_0 / den_0

        F_local = self.local_att(x.clone())

        top_T = 15  # The default maximum value of T is 15
        cosine_max, indexes = torch.topk(cosine, top_T, dim=2)
        dy_T = top_T
        for t in range(top_T):
            if torch.mean(cosine_max[:, :, t]) >= 0.5:
                dy_T = t
        dy_T = max(2, dy_T)

        mask = torch.ones(tmp.size(0), tmp.size(2) * tmp.size(3)).cuda()
        mask_index = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1)
        idx_b = torch.arange(tmp.size(0)).long().unsqueeze(1).expand(tmp.size(0), mask_index.size(1))

        rtn = tmp.clone().permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        for t in range(1, dy_T):
            mask_index_top = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1).gather(1, indexes[:, :, t])
            ind_1st_top = torch.zeros(tmp.size(0), tmp.size(2) * tmp.size(3), tmp.size(2) * tmp.size(3)).cuda()
            ind_1st_top[(idx_b, mask_index, mask_index_top)] = 1
            rtn += torch.bmm(ind_1st_top, former)
        rtn = rtn / dy_T
        F_global = rtn.permute(0, 2, 1).view(tmp.shape)
        # The following line maybe useful when the location of Attention() in the network is changed.
        # F_global = nn.UpsamplingNearest2d(size=(x.shape[2], x.shape[3]))(rtn.float())
        x = torch.cat([x, F_global, F_local], dim=1)
        return x


def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]


class dense_block(nn.Module):
    def __init__(self, in_channels, num_conv, kernel_size, filters, output_channels, dilate_rate, weight_decay, name,
                 down_sample, is_training, bn_in, strides, padding):
        super(dense_block, self).__init__()
        self.num_conv = num_conv
        # if(self.num_conv==2):
        #     self.conv1 = ConvX(in_channels,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv2 = ConvX(in_channels+filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        # if(self.num_conv==4):
        #     self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
        #                        is_training)
        #     self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
        #                        dilate_rate, is_training)
        #     self.conv3 = ConvX(in_channels+2*filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,dilate_rate, is_training)
        self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
                           is_training)
        self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv3 = ConvX(in_channels + 2 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.down_sample = down_sample
        if (self.num_conv == 2):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 2 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        if (self.num_conv == 3):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 3 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        if (self.num_conv == 4):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 4 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        self.se_layer = ChannelSELayer(output_channels, reduction_ratio=2)

    def forward(self, x):
        if (self.num_conv == 2):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x, conv1_output], dim=1)
            conv2_output = self.conv2(conv2_input)
            transition_input = torch.cat([x, conv1_output, conv2_output], dim=1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            x = self.se_layer(x)
            return x
        if (self.num_conv == 3):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x, conv1_output], dim=1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x, conv1_output, conv2_output], dim=1)
            conv3_output = self.conv3(conv3_input)
            transition_input = torch.cat([x, conv1_output, conv2_output, conv3_output], dim=1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            x = self.se_layer(x)
            return x
        if (self.num_conv == 4):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x, conv1_output], dim=1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x, conv1_output, conv2_output], dim=1)
            conv3_output = self.conv3(conv3_input)
            conv4_input = torch.cat([x, conv1_output, conv2_output, conv3_output], dim=1)
            conv4_output = self.conv4(conv4_input)
            transition_input = torch.cat([x, conv1_output, conv2_output, conv3_output, conv4_output], dim=1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            x = self.se_layer(x)
            return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()

        # 定义深度卷积操作
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        # 定义逐点卷积操作
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class NastNet(nn.Module):
    def __init__(self):
        bn_in = 'bn'
        super().__init__()
        self.combind = CombindConv2D(32)  # 此处填入数值n - 9（SRM） - 3（Bayer） 后是实际存在的卷积层个数
        self.conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.re = nn.ReLU(inplace=True)

        self.cellop1=nn.Identity()
        self.cellop2=dense_block(in_channels=64, num_conv=3, kernel_size=3, filters=32, output_channels=64,
                                 dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                 bn_in=bn_in, strides=1, padding=1)
        self.aftercell1= nn.Conv2d(128, 64, 3, 1, 1)
        self.cell2op1=dense_block(in_channels=64, num_conv=3, kernel_size=3, filters=32, output_channels=64,
                                 dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                 bn_in=bn_in, strides=1, padding=1)
        self.cell2op2=dense_block(in_channels=64, num_conv=3, kernel_size=3, filters=32, output_channels=64,
                                 dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                 bn_in=bn_in, strides=1, padding=1)
        self.aftercell2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.cell3op1=SeparableConv2D(64, 64, 3, 1, 1)

        self.cell3op2=nn.Conv2d(64, 64, 3, 1, 1)
        self.aftercell3 =nn.Conv2d(128, 64, 3, 1, 1)
        self.cell4op1= nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.cell4op2=SeparableConv2D(64, 64, 3, 1, 1)
        self.aftercell4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.cell5op1 =nn.Identity()
        self.cell5op2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.aftercell5 = nn.Conv2d(128, 64, 3, 2, 1)
        self.cell6op1 =SeparableConv2D(64, 64, 3, 1, 1)
        self.cell6op2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.aftercell6 = nn.Conv2d(128, 256, 3, 2, 1)
        self.cbam = nn.Sequential(
            MobileViTAttention(),
            nn.Conv2d(256, 64, 3, 1, 1)
        )
        #self.cbam = CBAMLayer(256)
        # 这里改掉
        self.nol2 = L2Norm()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # zoop去掉了
        self.cell7op1 = dense_block(in_channels=64, num_conv=3, kernel_size=3, filters=32, output_channels=64,
                                 dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                 bn_in=bn_in, strides=1, padding=1)
        self.cell7op2 =  nn.Identity()
        self.aftercell7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.cell8op1 =nn.Conv2d(64, 64, 3, 1, 1)
        self.cell8op2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.aftercell8 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.cell9op1 = dense_block(in_channels=64, num_conv=3, kernel_size=3, filters=32, output_channels=64,
                                 dilate_rate=1, weight_decay=0, name='', down_sample=False, is_training=True,
                                 bn_in=bn_in, strides=1, padding=1)
        self.cell9op2 = nn.Identity()
        self.aftercell9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.cell10op1 = nn.Conv2d(64, 64, 5, 1, 2)
        self.cell10op2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.aftercell10 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.decision = nn.Sequential(
            nn.Conv2d(64, 8, 7, 1, 3),
            nn.Conv2d(8, 1, 7, 1, 3)
        )

        self.sig = nn.Sigmoid()
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`

    def forward(self, x):
        x = self.combind(x)
        #4,5,7进行的缩小# x12 = torch.cat([x12, x5], axis=1)
        x = self.conv1(x)
        x = self.re(x)
        xc1p1=self.cellop1(x)
        xc1p2=self.cellop2(x)
        xc1 =self.aftercell1(torch.cat([xc1p1, xc1p2], axis=1))
        xc1=self.re(xc1)
        xc2p1 = self.cell2op1(x)
        xc2p2 = self.cell2op2(xc1)

        xc2 = self.aftercell2(torch.cat([xc2p1, xc2p2], axis=1))
        xc2 = self.re(xc2)
        xc3p1 = self.cell3op1(xc1)
        xc3p2 = self.cell3op2(xc3p1)
        xc3 = self.aftercell3(torch.cat([xc3p1, xc3p2], axis=1))
        xc3 = self.re(xc3)
        xc4p1 = self.cell4op1(xc2)
        xc4p2 = self.cell4op2(xc4p1)
        xc4 = self.aftercell4(torch.cat([xc4p1, xc4p2], axis=1))
        xc4 = self.re(xc4)
        xc5p1 = self.cell5op1(xc4)
        xc5p2 = self.cell5op2(xc4)
        xc5 = self.aftercell5(torch.cat([xc5p1, xc5p2], axis=1))
        xc5 = self.re(xc5)

        xc6p1 = self.cell6op1(xc5)
        xc6p2 = self.cell6op2(xc5)
        xc6 = self.aftercell6(torch.cat([xc6p1, xc6p2], axis=1))
        xc6 = self.re(xc6)

        xcbam=self.cbam(xc6)
        xcbam = self.nol2(xcbam)

        xc7p1 = self.cell7op1(xcbam)
        xc7p2 = self.cell7op2(xcbam)
        xc7 = self.aftercell7(torch.cat([xc7p1, xc7p2], axis=1))

        xc7=self.up(xc7)
        xc7 = self.re(xc7)

        xc8p1 = self.cell8op1(xc7)
        xc8p2 = self.cell8op2(xc5)

        xc8 = self.aftercell8(torch.cat([xc8p1, xc8p2], axis=1))
        xc8 = self.up(xc8)
        xc8 = self.re(xc8)

        xc9p1 = self.cell9op1(xc8)
        xc9p2 = self.cell9op2(xc8)
        xc9 = self.aftercell9(torch.cat([xc9p1, xc9p2], axis=1))
        xc9 = self.re(xc9)

        xc10p1 = self.cell10op1(xc9)
        xc10p2 = self.cell10op2(xc8)
        xc10 = self.aftercell10(torch.cat([xc10p1, xc10p2], axis=1))
        xc10 = self.re(xc10)

        x=self.decision(xc10)
        x = self.sig(x)
        # 256 64 64
        return x

        return x
x = torch.randn(1, 3, 256, 256).cuda()


model = NastNet().cuda()
#net=nn.Conv2d(256, 128,5, 1, 2)

y = model(x).cuda()
print(y.shape)