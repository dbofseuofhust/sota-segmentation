##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
affine_par = True

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

from .oc_module.resnet_block import conv3x3, Bottleneck
from .oc_module.base_oc_block import BaseOC_Module

torch_ver = torch.__version__[:3]

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # self.inplanes = 128
        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=False)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=False)
        # self.conv3 = conv3x3(64, 128)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU(inplace=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1)) # we do not apply multi-grid method here

        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            )
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]

        # x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        # x = self.maxpool(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.context(x)
        x = self.cls(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

def get_resnet101_base_oc_dsn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = ResNet(Bottleneck,[3, 4, 23, 3], datasets[dataset.lower()].NUM_CLASS)
    old_dict = model_zoo.load_url(model_urls[backbone])
    model_dict = model.state_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model
