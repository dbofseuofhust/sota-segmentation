import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models import BaseNet
import torchvision

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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

class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self,in_dim):
        super(PAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0, 2, 1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize,C,-1)
        proj_key = x.view(m_batchsize,C,-1).permute(0,2,1)
        energy = torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize,C,-1)

        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

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

class DecoderDAHeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            DANetHead(out_channels,out_channels),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class DecoderSCSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            SCSEBlock(out_channels),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)

class Unet(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(Unet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        # super(Unet, self).__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], nclass, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2) # [2, 1024, 32, 32]
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]]) # [2, 256, 32, 32]
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn,x])
        else:
            return x

class SCSEUnet(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(SCSEUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.layer1 = DecoderSCSEBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderSCSEBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderSCSEBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderSCSEBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderSCSEBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], nclass, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn,x])
        else:
            return x

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels))

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv
        x = self.conv8(feat_sum)

        return x

class DAHeadUnet(BaseNet):

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(DAHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add danet attention
        self.head = DANetHead(encoder_channels[0],encoder_channels[0])

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

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

class OCHeadUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,oc_arch='asp',
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(OCHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.oc_arch = oc_arch
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

class SCSEHCOCHeadUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,oc_arch='asp',
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(SCSEHCOCHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.oc_arch = oc_arch
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.layer1 = DecoderSCSEBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderSCSEBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderSCSEBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderSCSEBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderSCSEBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.hc = nn.Sequential(nn.Conv2d(out_channels[1]+out_channels[2]+out_channels[3]+out_channels[4], out_channels[4], kernel_size=3, padding=1),
                                   nn.ELU(True))

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        outputs = []
        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]]) # 128,64,64
        outputs.append(F.upsample(x, scale_factor=8, mode='bilinear', align_corners=True))
        x = self.layer3([x, skips[2]]) # 64,128,128
        outputs.append(F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True))
        x = self.layer4([x, skips[3]]) # 32,256,256
        outputs.append(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = self.layer5([x, None]) # 16,512,512
        outputs.append(x)
        x = torch.cat(outputs,dim=1)
        x = self.hc(x)
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

class HCOCHeadUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,oc_arch='asp',
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(HCOCHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.oc_arch = oc_arch
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.hc = nn.Sequential(nn.Conv2d(out_channels[1]+out_channels[2]+out_channels[3]+out_channels[4], out_channels[4], kernel_size=3, padding=1),
                                   nn.ELU(True))

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        outputs = []
        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]]) # 128,64,64
        outputs.append(F.upsample(x, scale_factor=8, mode='bilinear', align_corners=True))
        x = self.layer3([x, skips[2]]) # 64,128,128
        outputs.append(F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True))
        x = self.layer4([x, skips[3]]) # 32,256,256
        outputs.append(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = self.layer5([x, None]) # 16,512,512
        outputs.append(x)
        x = torch.cat(outputs,dim=1)
        x = self.hc(x)
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

class SCSEOCHeadUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,oc_arch='asp',
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(SCSEOCHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.oc_arch = oc_arch
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.layer1 = DecoderSCSEBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderSCSEBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderSCSEBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderSCSEBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderSCSEBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

class HCSCSEUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(HCSCSEUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        self.layer1 = DecoderSCSEBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderSCSEBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderSCSEBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderSCSEBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderSCSEBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # ad Hyper Column part
        self.hc = nn.Sequential(nn.Conv2d(out_channels[1]+out_channels[2]+out_channels[3]+out_channels[4], out_channels[4], kernel_size=3, padding=1),
                                   nn.ELU(True))
        self.final_conv = nn.Conv2d(out_channels[4], nclass, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        outputs = []
        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]]) # 128,64,64
        outputs.append(F.upsample(x, scale_factor=8, mode='bilinear', align_corners=True))
        x = self.layer3([x, skips[2]]) # 64,128,128
        outputs.append(F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True))
        x = self.layer4([x, skips[3]]) # 32,256,256
        outputs.append(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = self.layer5([x, None]) # 16,512,512
        outputs.append(x)
        x = torch.cat(outputs,dim=1)
        x = self.hc(x)
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

class OCDAHeadUnet(BaseNet):

    def __init__(self,nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, center=False,oc_arch='asp',
                 encoder_channels=None, decoder_channels=(256, 128, 64, 32, 16), use_batchnorm=True, **kwargs):
        super(OCDAHeadUnet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.oc_arch = oc_arch
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        # add ocnet attention
        self.head = OCHead(in_ch=encoder_channels[0],nclass=encoder_channels[0],oc_arch=self.oc_arch)

        self.layer1 = DecoderDAHeadBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderDAHeadBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderDAHeadBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderDAHeadBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderDAHeadBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.final_conv = nn.Conv2d(out_channels[4],nclass,kernel_size=(1,1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        size = x.size()[2:]

        x0 = self.pretrained.conv1(x)
        x0 = self.pretrained.bn1(x0)
        x0 = self.pretrained.relu(x0)
        x1 = self.pretrained.maxpool(x0)
        x1 = self.pretrained.layer1(x1)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x_dsn = self.dsn(x3)
        x4 = self.pretrained.layer4(x3)

        x = [x4, x3, x2, x1, x0]

        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        encoder_head = self.head(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn, x])
        else:
            return x

def get_unet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = Unet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, encoder_channels=encoder_channels_dict[backbone],**kwargs)
    # model = Unet(nclass=6, backbone=backbone, root=root, encoder_channels=encoder_channels_dict[backbone], **kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_scseunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', **kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = SCSEUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], **kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_hcscseunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', **kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = HCSCSEUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], **kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_daheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', **kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DAHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], **kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_ocheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',**kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = OCHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], oc_arch=oc_arch,**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_scseocheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',**kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = SCSEOCHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], oc_arch=oc_arch,**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_hcocheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',**kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = HCOCHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], oc_arch=oc_arch,**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_scsehcocheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',**kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = SCSEHCOCHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], oc_arch=oc_arch,**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

def get_scsehcocheadunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
             root='./pretrain_models', oc_arch='asp',**kwargs):
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
    }

    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = SCSEHCOCHeadUnet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root,
                 encoder_channels=encoder_channels_dict[backbone], oc_arch=oc_arch,**kwargs)

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

if __name__ == '__main__':

    model = get_unet(backbone='resnet50')
    img = torch.randn(2,3,512,512)
    out = model(img)