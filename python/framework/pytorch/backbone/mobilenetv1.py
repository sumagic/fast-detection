import torch as tr
import torch.nn as nn
import glog
import sys
import numpy as np

from framework.pytorch.util.conv2d import DepthWiseConv2d

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1_s2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.dw2_s1 = DepthWiseConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3_s1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw4_s2 = DepthWiseConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, bias=True)
        self.conv5_s1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw6_s1 = DepthWiseConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv7_s1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw8_s2 = DepthWiseConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, bias=True)
        self.conv9_s1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw10_s1 = DepthWiseConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv11_s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw12_s2 = DepthWiseConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, bias=True)
        self.conv13_s1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw_common_s1 = DepthWiseConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_common_s1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw24_s2 = DepthWiseConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, bias=True)
        self.conv25_s1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dw26_s2 = DepthWiseConv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=4, dilation=1, bias=True)
        self.conv27_s1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.avgpool28_s1 = nn.AvgPool2d((7, 7), stride=(1, 1))
        self.linear29_s1 = nn.Linear(1024, 1000)
        self.softmax30_s1 = nn.Softmax(dim=0)



    def forward(self, x):
        x = self.conv1_s2(x)
        x = self.dw2_s1(x)
        x = self.conv3_s1(x)
        x = self.dw4_s2(x)
        x = self.conv5_s1(x)
        x = self.dw6_s1(x)
        x = self.conv7_s1(x)
        x = self.dw8_s2(x)
        x = self.conv9_s1(x)
        x = self.dw10_s1(x)
        x = self.conv11_s1(x)
        x = self.dw12_s2(x)
        x = self.conv13_s1(x)
        loop = 5
        for i in range(loop):
            x = self.dw_common_s1(x)
            x = self.conv_common_s1(x)
        x = self.dw24_s2(x)
        x = self.conv25_s1(x)
        x = self.dw26_s2(x)
        x = self.conv27_s1(x)
        x = self.avgpool28_s1(x)
        x = x.view(1, 1024)
        x = self.linear29_s1(x)
        x = self.softmax30_s1(x)
        cmax = tr.argmax(x, dim=1)

        return cmax