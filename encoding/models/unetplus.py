from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from ..models import BaseNet
import torchvision
import torch.nn.functional as F


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch,scse=False):
        super(DecoderBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.scse = scse

        if self.scse:
            self.scse_block = SCSEBlock(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        if self.scse:
            output = self.scse_block(output)

        return output

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = x * chn_se

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se
        return chn_se + spa_se

class BaseAttentionBlock(nn.Module):
    """The basic implementation for self-attention block/non-local block."""

    def __init__(self, in_channels, out_channels, key_channels, value_channels,
                 scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BaseAttentionBlock, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(scale)

        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels),
            nn.ReLU(True)
        )
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.bmm(query, key) * (self.key_channels ** -.5)
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(w, h), mode='bilinear', align_corners=True)

        return context

class BaseOCModule(nn.Module):
    """Base-OC"""

    def __init__(self, in_channels, out_channels, key_channels, value_channels,
                 scales=([1]), norm_layer=nn.BatchNorm2d, concat=True, **kwargs):
        super(BaseOCModule, self).__init__()
        self.stages = nn.ModuleList([
            BaseAttentionBlock(in_channels, out_channels, key_channels, value_channels, scale, norm_layer, **kwargs)
            for scale in scales])
        in_channels = in_channels * 2 if concat else in_channels
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.05)
        )
        self.concat = concat

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        if self.concat:
            context = torch.cat([context, x], 1)
        out = self.project(context)
        return out

class PyramidAttentionBlock(nn.Module):
    """The basic implementation for pyramid self-attention block/non-local block"""

    def __init__(self, in_channels, out_channels, key_channels, value_channels,
                 scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidAttentionBlock, self).__init__()
        self.scale = scale
        self.value_channels = value_channels
        self.key_channels = key_channels

        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels),
            nn.ReLU(True)
        )
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()

        local_x = list()
        local_y = list()
        step_w, step_h = w // self.scale, h // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = step_w * i, step_h * j
                end_x, end_y = min(start_x + step_w, w), min(start_y + step_h, h)
                if i == (self.scale - 1):
                    end_x = w
                if j == (self.scale - 1):
                    end_y = h
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = list()
        local_block_cnt = (self.scale ** 2) * 2
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]

            w_local, h_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1).permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1).permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local) * (self.key_channels ** -.5)
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(sim_map, value_local).permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, w_local, h_local)
            local_list.append(context_local)

        context_list = list()
        for i in range(0, self.scale):
            row_tmp = list()
            for j in range(self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = self.W(context)

        return context

class PyramidOCModule(nn.Module):
    """Pyramid-OC"""

    def __init__(self, in_channels, out_channels, key_channels, value_channels,
                 scales=([1]), norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidOCModule, self).__init__()
        self.stages = nn.ModuleList([
            PyramidAttentionBlock(in_channels, out_channels, key_channels, value_channels, scale, norm_layer, **kwargs)
            for scale in scales])
        self.up_dr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * len(scales), 1),
            norm_layer(in_channels * len(scales)),
            nn.ReLU(True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * len(scales) * 2, out_channels, 1),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.05)
        )

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = [self.up_dr(x)]
        for i in range(len(priors)):
            context += [priors[i]]
        context = torch.cat(context, 1)
        out = self.project(context)
        return out

class ASPOCModule(nn.Module):
    """ASP-OC"""

    def __init__(self, in_channels, out_channels, key_channels, value_channels,
                 atrous_rates=(12, 24, 36), norm_layer=nn.BatchNorm2d, **kwargs):
        super(ASPOCModule, self).__init__()
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(True),
            BaseOCModule(out_channels, out_channels, key_channels, value_channels, ([2]), norm_layer, False, **kwargs))

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate1, dilation=rate1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate2, dilation=rate2, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate3, dilation=rate3, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        feat1 = self.context(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.project(out)
        return out

class UNetPlus(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, use_batchnorm=True,scse=True,**kwargs):
        super(UNetPlus, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.inplanes = 64

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_channels = encoder_channels[::-1]

        self.x01 = DecoderBlock(in_channels[0] + in_channels[1], in_channels[0], in_channels[0],scse=scse)
        self.x11 = DecoderBlock(in_channels[1] + in_channels[2], in_channels[1], in_channels[1],scse=scse)
        self.x21 = DecoderBlock(in_channels[2] + in_channels[3], in_channels[2], in_channels[2],scse=scse)

        self.x02 = DecoderBlock(in_channels[0] * 2 + in_channels[1], in_channels[0], in_channels[0],scse=scse)
        self.x12 = DecoderBlock(in_channels[1] * 2 + in_channels[2], in_channels[1], in_channels[1],scse=scse)

        self.x03 = DecoderBlock(in_channels[0] * 3 + in_channels[1], in_channels[0], in_channels[0],scse=scse)

        self.x31 = DecoderBlock(in_channels[3] + in_channels[4], in_channels[3], in_channels[3],scse=scse)
        self.x22 = DecoderBlock(in_channels[2] * 2 + in_channels[3], in_channels[2], in_channels[2],scse=scse)
        self.x13 = DecoderBlock(in_channels[1] * 3 + in_channels[2], in_channels[1], in_channels[1],scse=scse)
        self.x04 = DecoderBlock(in_channels[0] * 4 + in_channels[1], in_channels[0], in_channels[0],scse=scse)

        # dsn
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.final_conv = nn.Conv2d(in_channels[0], nclass, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x00 = self.relu(x)

        # x10 = self.maxpool(x00)
        # x10 = self.layer1(x10)

        # x20 = self.layer2(x10)
        # x30 = self.layer3(x20)
        # x40 = self.layer4(x30)

        size = x.size()[2:]

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x00 = self.pretrained.relu(x)

        x10 = self.pretrained.maxpool(x00)
        x10 = self.pretrained.layer1(x10)

        x20 = self.pretrained.layer2(x10)
        x30 = self.pretrained.layer3(x20)
        x_dsn = self.dsn(x30)
        x40 = self.pretrained.layer4(x30)

        x01 = self.x01(torch.cat([x00, self.UP(x10)], 1))

        x11 = self.x11(torch.cat([x10, self.UP(x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.UP(x11)], 1))

        x21 = self.x21(torch.cat([x20, self.UP(x30)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.UP(x21)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.UP(x12)], 1))

        x31 = self.x31(torch.cat([x30, self.UP(x40)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.UP(x31)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.UP(x22)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.UP(x13)], 1))

        output = self.final_conv(x04)
        output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        
        if self.training:
            return tuple([x_dsn,output])
        else:
            return output

class OCHead(nn.Module):
    def __init__(self, in_ch, nclass, oc_arch, norm_layer=nn.BatchNorm2d, **kwargs):
        super(OCHead, self).__init__()
        if oc_arch == 'base':
            self.context = nn.Sequential(
                nn.Conv2d(in_ch, 512, 3, 1, padding=1, bias=False),
                norm_layer(512),
                nn.ReLU(True),
                BaseOCModule(512, 512, 256, 256, scales=([1]), norm_layer=norm_layer, **kwargs))
        elif oc_arch == 'pyramid':
            self.context = nn.Sequential(
                nn.Conv2d(in_ch, 512, 3, 1, padding=1, bias=False),
                norm_layer(512),
                nn.ReLU(True),
                PyramidOCModule(512, 512, 256, 512, scales=([1, 2, 3, 6]), norm_layer=norm_layer, **kwargs))
        elif oc_arch == 'asp':
            self.context = ASPOCModule(in_ch, 512, 256, 512, norm_layer=norm_layer, **kwargs)
        else:
            raise ValueError("Unknown OC architecture!")

        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.context(x)
        return self.out(x)

class OCHeadUNetPlus(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, use_batchnorm=True,scse=True,oc_arch='asp',**kwargs):
        super(OCHeadUNetPlus, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.inplanes = 64
        self.oc_arch = oc_arch

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_channels = encoder_channels[::-1]

        self.x01 = DecoderBlock(in_channels[0] + in_channels[1], in_channels[0], in_channels[0],scse=scse)
        self.x11 = DecoderBlock(in_channels[1] + in_channels[2], in_channels[1], in_channels[1],scse=scse)
        self.x21 = DecoderBlock(in_channels[2] + in_channels[3], in_channels[2], in_channels[2],scse=scse)

        self.x02 = DecoderBlock(in_channels[0] * 2 + in_channels[1], in_channels[0], in_channels[0],scse=scse)
        self.x12 = DecoderBlock(in_channels[1] * 2 + in_channels[2], in_channels[1], in_channels[1],scse=scse)

        self.x03 = DecoderBlock(in_channels[0] * 3 + in_channels[1], in_channels[0], in_channels[0],scse=scse)

        self.x31 = DecoderBlock(in_channels[3] + in_channels[4], in_channels[3], in_channels[3],scse=scse)
        self.x22 = DecoderBlock(in_channels[2] * 2 + in_channels[3], in_channels[2], in_channels[2],scse=scse)
        self.x13 = DecoderBlock(in_channels[1] * 3 + in_channels[2], in_channels[1], in_channels[1],scse=scse)
        self.x04 = DecoderBlock(in_channels[0] * 4 + in_channels[1], in_channels[0], in_channels[0],scse=scse)

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)

        # dsn
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.final_conv = nn.Conv2d(in_channels[0], nclass, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x00 = self.relu(x)

        # x10 = self.maxpool(x00)
        # x10 = self.layer1(x10)

        # x20 = self.layer2(x10)
        # x30 = self.layer3(x20)
        # x40 = self.layer4(x30)

        size = x.size()[2:]

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x00 = self.pretrained.relu(x)

        x10 = self.pretrained.maxpool(x00)
        x10 = self.pretrained.layer1(x10)

        x20 = self.pretrained.layer2(x10)
        x30 = self.pretrained.layer3(x20)
        x_dsn = self.dsn(x30)
        x40 = self.pretrained.layer4(x30)

        x40 = self.head(x40)

        x01 = self.x01(torch.cat([x00, self.UP(x10)], 1))

        x11 = self.x11(torch.cat([x10, self.UP(x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.UP(x11)], 1))

        x21 = self.x21(torch.cat([x20, self.UP(x30)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.UP(x21)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.UP(x12)], 1))

        x31 = self.x31(torch.cat([x30, self.UP(x40)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.UP(x31)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.UP(x22)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.UP(x13)], 1))

        output = self.final_conv(x04)
        output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        
        if self.training:
            return tuple([x_dsn,output])
        else:
            return output

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

# def get_pose_net(cfg, is_train, **kwargs):
#     num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
#     block_class, layers = resnet_spec[num_layers]
#     model = PoseUNet(block_class, layers, cfg, encoder_channels=(2048, 1024, 512, 256, 64), **kwargs)

#     if is_train and cfg.MODEL.INIT_WEIGHTS:
#         model.init_weights(cfg.MODEL.PRETRAINED)
#     return model

def get_unetplus(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',is_dilated=False,**kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }

    encoder_channels_dict = {
        'resnet18': (512, 256, 128, 64, 64),
        'resnet34': (512, 256, 128, 64, 64),
        'resnet50': (2048, 1024, 512, 256, 64),
        'resnet101': (2048, 1024, 512, 256, 64),
        'resnet152': (2048, 1024, 512, 256, 64),
        'atrous_resnet50': (2048, 1024, 512, 256, 64),
        'atrous_resnet101': (2048, 1024, 512, 256, 64),
        'atrous_resnet152': (2048, 1024, 512, 256, 64),
        'resnet101_ibn_a': (2048, 1024, 512, 256, 64),
        'resnext101_ibn_a': (2048, 1024, 512, 256, 64),
        'se_resnet18': (512, 256, 128, 64, 64),
        'se_resnet34': (512, 256, 128, 64, 64),
        'se_resnet50': (2048, 1024, 512, 256, 64),
        'se_resnet101': (2048, 1024, 512, 256, 64),
        'se_resnet152': (2048, 1024, 512, 256, 64),
        'se_resnext50_32x4d': (2048, 1024, 512, 256, 64),
        'se_resnext101_32x4d': (2048, 1024, 512, 256, 64),
        'resnext101_32x4d': (2048, 1024, 512, 256, 64),
        'resnext101_64x4d': (2048, 1024, 512, 256, 64),
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = UNetPlus(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone],**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_ocheadunetplus(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',is_dilated=False,**kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }

    encoder_channels_dict = {
        'resnet18': (512, 256, 128, 64, 64),
        'resnet34': (512, 256, 128, 64, 64),
        'resnet50': (2048, 1024, 512, 256, 64),
        'resnet101': (2048, 1024, 512, 256, 64),
        'resnet152': (2048, 1024, 512, 256, 64),
        'atrous_resnet50': (2048, 1024, 512, 256, 64),
        'atrous_resnet101': (2048, 1024, 512, 256, 64),
        'atrous_resnet152': (2048, 1024, 512, 256, 64),
        'resnet101_ibn_a': (2048, 1024, 512, 256, 64),
        'resnext101_ibn_a': (2048, 1024, 512, 256, 64),
        'se_resnet18': (512, 256, 128, 64, 64),
        'se_resnet34': (512, 256, 128, 64, 64),
        'se_resnet50': (2048, 1024, 512, 256, 64),
        'se_resnet101': (2048, 1024, 512, 256, 64),
        'se_resnet152': (2048, 1024, 512, 256, 64),
        'se_resnext50_32x4d': (2048, 1024, 512, 256, 64),
        'se_resnext101_32x4d': (2048, 1024, 512, 256, 64),
        'resnext101_32x4d': (2048, 1024, 512, 256, 64),
        'resnext101_64x4d': (2048, 1024, 512, 256, 64),
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = OCHeadUNetPlus(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone],**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model


if __name__ == '__main__':

    num_layers = 152

    block_class, layers = resnet_spec[num_layers]

    model = PoseUNet(block_class, layers, cfg=None, encoder_channels=(2048, 1024, 512, 256, 64))

    img = torch.randn(2,3,384,288)
    out = model(img)
    print(out.size())