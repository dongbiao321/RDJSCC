# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/17 14:43
@Auth ： dongb
@File ：CVIE.py
@IDE ：PyCharm
"""

import torch.nn as nn
import torch
import torch.fft

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class CCF_fuse(nn.Module):
    def __init__(self, Nin, Nh, No, config):
        super(CCF_fuse, self).__init__()
        self.fc1 = nn.Linear(Nin , Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.temp = 5
        self.config = config
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y, SNR1, SNR2):
        corNum = torch.nn.functional.cosine_similarity(x, y, dim=3)
        corNum = torch.pow(corNum, 2)
        corNum = torch.mean(corNum, [2])
        snr1 = SNR1.to(self.config["device"])
        snr2 = SNR2.to(self.config["device"])
        out = torch.cat((corNum, snr1, snr2), 1).to(self.config["device"])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        soft = self.softmax(out)
        return soft

class CVIE(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1,config=None):
        super(CVIE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.head_dim = self.out_planes // self.head
        self.config = config

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv1 = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=2,
                                  stride=stride)
        self.dep_conv2 = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=2,
                                  stride=stride)

        self.reset_parameters()
        self.CCF_fuse = CCF_fuse(in_planes + 2, (in_planes + 2)//2, 2, self.config)

    def reset_parameters(self):
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv1.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv1.bias = init_rate_0(self.dep_conv1.bias)
        self.dep_conv2.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv2.bias = init_rate_0(self.dep_conv2.bias)

    def forward(self, x, y, SNR1, SNR2):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        b, c, h, w = q.shape

        ## CVIE
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_att = self.dep_conv1(f_conv)

        ## CVIE
        q = self.conv1(y)
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv2(f_conv)

        ## CCF
        self.soft = self.CCF_fuse(x, y, SNR1, SNR2)
        rate1 = self.soft[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        rate2 = self.soft[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return rate1 * out_att + rate2 * out_conv, self.soft[:,0].mean(0).item(), self.soft[:,1].mean(0).item()
