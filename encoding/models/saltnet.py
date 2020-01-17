import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ..dilated import IBN
from ..models import BaseNet
import torch.utils.model_zoo as model_zoo


model_urls = {
    'atrous_resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'atrous_resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'atrous_resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'atrous_resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'atrous_resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'atrous_se_resnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'atrous_se_resnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
    'atrous_se_resnet152': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
    'atrous_se_resnext50_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'atrous_se_resnext101_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
    'atrous_senet154': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'
}

def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

def conv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)

def conv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)

def conv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)

class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class ConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, norm_type = None):
        super(ConvRelu,self).__init__()

        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False

        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True

        if kernel_size == 3:
            self.conv = conv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = conv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = conv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = conv1x1(in_, out, is_bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x

class ImprovedIBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ImprovedIBNaDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = IBN(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
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
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):
    def __init__(self,in_channels, channels, out_channels,is_attention=False,is_dilated=False,F_g=None, F_l=None, F_int=None):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

        self.is_attention = is_attention
        if self.is_attention:
            assert F_g is not None
            assert F_l is not None
            assert F_int is not None
            self.attb = AttentionBlock(F_g=F_g, F_l=F_l, F_int=F_int)

        self.is_dilated = is_dilated

    def forward(self, x, e = None):
        if not self.is_dilated:
            x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:

            if self.is_attention:
                e = self.attb(g=x,x=e)

            x = torch.cat([x,e],1)
            x = F.dropout2d(x, p = 0.50)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(Bottleneck, self).__init__()
        planes = inplanes // 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)

        self.is_skip = True
        if inplanes != outplanes:
            self.is_skip = False

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

        if self.is_skip:
            out += residual
        out = self.relu(out)

        return out

class DecoderBottleneck(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(DecoderBottleneck, self).__init__()

        self.block1 = Bottleneck(in_channels, channels)
        self.block2 = Bottleneck(channels, out_channels)
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.SCSE(x)
        return x

# class SaltNet34(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None, is_hypercol=False, **kwargs):
#         assert backbone == 'resnet34'
#         super(SaltNet34, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         # self.num_classes = num_classes
#         self.nclass = nclass
#
#         self.is_hypercol = is_hypercol
#
#         # ori
#         # self.encoder = torchvision.models.resnet34(pretrained=True)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.conv1 = nn.Sequential(self.encoder.conv1,
#         #                            self.encoder.bn1,
#         #                            self.encoder.relu)
#         # self.conv2 = self.encoder.layer1
#         # self.conv3 = self.encoder.layer2
#         # self.conv4 = self.encoder.layer3
#         # self.conv5 = self.encoder.layer4
#         # self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
#         # self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
#         # self.center_fc = nn.Linear(64, self.num_classes)
#
#         # modify by db to unify with base.py
#         self.encoder = self.pretrained
#         self.conv1 = nn.Sequential(self.encoder.conv1,
#                                    self.encoder.bn1,
#                                    self.encoder.relu)
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
#         self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
#
#         self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(inplace=True),
#                                     nn.MaxPool2d(kernel_size=2,stride=2))
#
#         self.decoder5 = Decoder(256 + 512, 512, 64)
#         self.decoder4 = Decoder(64 + 256, 256, 64)
#         self.decoder3 = Decoder(64 + 128, 128, 64)
#         self.decoder2 = Decoder(64 + 64, 64, 64)
#         self.decoder1 = Decoder(64, 32, 64)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # ori
#         # self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
#         #                             nn.ReLU(inplace=True),
#         #                             nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#
#         if self.is_hypercol:
#             self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
#                                              nn.ReLU(inplace=True),
#                                              nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#         else:
#             self.logits_final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#
#         conv1 = self.conv1(x)     # 1/4
#         conv2 = self.conv2(conv1) # 1/4
#         conv3 = self.conv3(conv2) # 1/8
#         conv4 = self.conv4(conv3) # 1/16
#         conv5 = self.conv5(conv4) # 1/32
#
#         # ori
#         # center_512 = self.center_global_pool(conv5)
#         # center_64 = self.center_conv1x1(center_512)
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         center_512 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_512)
#
#         f = self.center(conv5)
#         d5 = self.decoder5(f, conv5)
#         d4 = self.decoder4(d5, conv4)
#         d3 = self.decoder3(d4, conv3)
#         d2 = self.decoder2(d3, conv2)
#         d1 = self.decoder1(d2)
#
#         if self.is_hypercol:
#             hypercol = torch.cat((
#                 d1,
#                 F.upsample(d2, scale_factor=2, mode='bilinear'),
#                 F.upsample(d3, scale_factor=4, mode='bilinear'),
#                 F.upsample(d4, scale_factor=8, mode='bilinear'),
#                 F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
#             hypercol = F.dropout2d(hypercol, p=0.50)
#             hypercol_add_center = torch.cat((
#                 hypercol,
#                 F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
#             x_final = self.logits_final(hypercol_add_center)
#         else:
#             x_final = self.logits_final(d1)
#
#         x_dsn = self.dsn(d4)
#         x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final

class SaltNetSRX50(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None,
                 is_hypercol=False,is_aspp=True,is_dilated=True,is_attention=False,global_context=False,**kwargs):
        super(SaltNetSRX50, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        # super(SaltNetSRX50, self).__init__()

        self.nclass = nclass
        self.encoder = self.pretrained

        # self.encoder = eval('torchvision.models.{}'.format(backbone))(False)
        # self.encoder = get_atrous_resnet('resnet50',16)

        self.is_hypercol = is_hypercol
        self.is_attention = is_attention
        self.is_aspp = is_aspp
        self.global_context = global_context
        self.is_dilated = is_dilated

        if 'atrous' in backbone:

            if not 'atrous_se' in backbone:
                old_dict = model_zoo.load_url(model_urls[backbone])
                model_dict = self.encoder.state_dict()
                old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
                model_dict.update(old_dict)
                self.encoder.load_state_dict(model_dict)
                print('loading {} imagenet pretrained weights done!'.format(backbone))

                self.conv1 = nn.Sequential(self.encoder.conv1,
                                               self.encoder.bn1,
                                               self.encoder.relu)
            else:
                self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                           self.encoder.layer0.bn1,
                                           self.encoder.layer0.relu1)
        else:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                       self.encoder.layer0.bn1,
                                       self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
        # ori
        # self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2)
                                    )

        self.decoder5 = Decoder(256 + 512*4, 512, 64, is_attention=self.is_attention,F_g=512 * 4, F_l=256, F_int=256)
        self.decoder4 = Decoder(64 + 256*4, 256, 64, is_attention=self.is_attention,F_g=256 * 4, F_l=64, F_int=64,is_dilated=self.is_dilated)
        self.decoder3 = Decoder(64 + 128*4, 128, 64, is_attention=self.is_attention,F_g=128 * 4, F_l=64, F_int=64)
        self.decoder2 = Decoder(64 + 64*4, 64, 64, is_attention=self.is_attention,F_g=64 * 4, F_l=64, F_int=64)
        self.decoder1 = Decoder(64, 32, 64)

        self.dsn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # ori
        # self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(64, 1, kernel_size=1, padding=0))

        if self.is_hypercol:
            self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
        else:
            self.logits_final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))

        if self.is_aspp:
            # parameters
            output_stride,indim,aspp_outdim,shortcut_kernel,input_channel = 32,256,256,1,2048

            if self.global_context:
                shortcut_dim = 48
            else:
                shortcut_dim = 0

            # to avoid overfitting
            self.dropout1 = nn.Dropout(0.5)
            self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=output_stride // 4)

            self.aspp = ASPP(dim_in=input_channel, dim_out=aspp_outdim, rate=32 // output_stride)

            self.center = nn.Sequential(
                nn.Conv2d(aspp_outdim+shortcut_dim, aspp_outdim, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(aspp_outdim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(aspp_outdim, aspp_outdim, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(aspp_outdim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            if self.global_context:
                self.shortcut_conv = nn.Sequential(
                    nn.Conv2d(indim, shortcut_dim, shortcut_kernel, 1, padding=shortcut_kernel // 2, bias=True),
                    nn.BatchNorm2d(shortcut_dim),
                    nn.ReLU(inplace=True),
                )

    def forward(self, x):

        h,w = x.size()[2],x.size()[3]

        conv1 = self.conv1(x)     # 1/4
        conv2 = self.conv2(conv1) # 1/4
        conv3 = self.conv3(conv2) # 1/8
        conv4 = self.conv4(conv3) # 1/16
        conv5 = self.conv5(conv4) # 1/32


        if self.is_aspp:
            aspp = self.aspp(conv5)
            aspp = self.dropout1(aspp)
            if self.global_context:
                short_cut = nn.AdaptiveAvgPool2d((h//16,w//16))(conv2)
                short_cut = self.shortcut_conv(short_cut)
                aspp = torch.cat([aspp,short_cut],dim=1)
            f = self.center(aspp)
        else:
            f = self.center(conv5)

        # ori
        # center_64_flatten = center_64.view(center_64.size(0), -1)
        # center_fc = self.center_fc(center_64_flatten)

        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        if self.is_hypercol:
            center_512 = self.center_global_pool(conv5)
            center_64 = self.center_conv1x1(center_512)
            hypercol = torch.cat((
                d1,
                F.upsample(d2, scale_factor=2, mode='bilinear'),
                F.upsample(d3, scale_factor=4, mode='bilinear'),
                F.upsample(d4, scale_factor=8, mode='bilinear'),
                F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
            hypercol = F.dropout2d(hypercol, p=0.50)
            hypercol_add_center = torch.cat((
                hypercol,
                F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
            x_final = self.logits_final(hypercol_add_center)
        else:
            x_final = self.logits_final(d1)

        x_dsn = self.dsn(d4)
        x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')

        if self.training:
            return tuple([x_dsn,x_final])
        else:
            return x_final

# class SaltNetSlimSRX50(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None, is_hypercol=False, **kwargs):
#         assert backbone == 'se_resnext50_32x4d'
#         super(SaltNetSlimSRX50, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         self.nclass = nclass
#         self.encoder = self.pretrained
#
#         self.is_hypercol = is_hypercol
#
#         self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
#                                    self.encoder.layer0.bn1,
#                                    self.encoder.layer0.relu1)
#
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
#         self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
#         # ori
#         # self.center_fc = nn.Linear(64, 2)
#
#         self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(inplace=True),
#                                     nn.MaxPool2d(kernel_size=2,stride=2))
#
#         self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True))
#         self.decoder5 = DecoderBottleneck(256 + 512, 512, 64)
#
#         self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True))
#         self.decoder4 = DecoderBottleneck(64 + 256, 256, 64)
#
#         self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
#         self.decoder3 = DecoderBottleneck(64 + 128, 128, 64)
#
#         self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
#         self.decoder2 = DecoderBottleneck(64 + 64, 64, 64)
#
#         self.decoder1 = DecoderBottleneck(64, 32, 64)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # ori
#         # self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
#         #                             nn.ReLU(inplace=True),
#         #                             nn.Conv2d(64, 1, kernel_size=1, padding=0))
#
#         if self.is_hypercol:
#             self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#         else:
#             self.logits_final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#
#         conv1 = self.conv1(x)     # 1/4
#         conv2 = self.conv2(conv1) # 1/4
#         conv3 = self.conv3(conv2) # 1/8
#         conv4 = self.conv4(conv3) # 1/16
#         conv5 = self.conv5(conv4) # 1/32
#
#         center_512 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_512)
#         # ori
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         f = self.center(conv5)
#
#         conv5 = self.dec5_1x1(conv5)
#         d5 = self.decoder5(f, conv5)
#
#         conv4 = self.dec4_1x1(conv4)
#         d4 = self.decoder4(d5, conv4)
#
#         conv3 = self.dec3_1x1(conv3)
#         d3 = self.decoder3(d4, conv3)
#
#         conv2 = self.dec2_1x1(conv2)
#         d2 = self.decoder2(d3, conv2)
#
#         d1 = self.decoder1(d2)
#
#         if self.is_hypercol:
#             hypercol = torch.cat((
#                 d1,
#                 F.upsample(d2, scale_factor=2, mode='bilinear'),
#                 F.upsample(d3, scale_factor=4, mode='bilinear'),
#                 F.upsample(d4, scale_factor=8, mode='bilinear'),
#                 F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
#             hypercol = F.dropout2d(hypercol, p=0.50)
#
#             hypercol_add_center = torch.cat((
#                 hypercol,
#                 F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
#             x_final = self.logits_final(hypercol_add_center)
#         else:
#             x_final = self.load_state_dict(d1)
#
#         x_dsn = self.dsn(d4)
#         x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final
#
# class SaltNetRXIBNA101(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None, **kwargs):
#         assert backbone == 'resnext101_ibn_a'
#         super(SaltNetRXIBNA101, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         self.nclass = nclass
#
#         num_filters = 32
#         # baseWidth = 4
#         # cardinality = 32
#
#         self.encoder = self.pretrained
#
#         self.conv1 = nn.Sequential(self.encoder.conv1,
#                                    self.encoder.bn1,
#                                    self.encoder.relu)
#
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
#         self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
#         # ori
#         # self.center_fc = nn.Linear(64, 2)
#
#         self.center_se = SELayer(512*4)
#         self.center = ImprovedIBNaDecoderBlock(512*4,  num_filters * 8)
#
#         self.dec5_se = SELayer(512*4 + num_filters * 8)
#         self.dec5 = ImprovedIBNaDecoderBlock(512*4 + num_filters * 8, num_filters * 8)
#
#         self.dec4_se = SELayer(256*4 + num_filters * 8)
#         self.dec4 = ImprovedIBNaDecoderBlock(256*4 + num_filters * 8, num_filters * 8)
#
#         self.dec3_se = SELayer(128*4 + num_filters * 8)
#         self.dec3 = ImprovedIBNaDecoderBlock(128*4 + num_filters * 8, num_filters * 4)
#
#         self.dec2_se = SELayer(64*4 + num_filters * 4)
#         self.dec2 = ImprovedIBNaDecoderBlock(64*4 + num_filters * 4, num_filters * 4)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(num_filters * 8, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # ori
#         # self.logits_no_empty = nn.Sequential(StConvRelu(num_filters * 4, num_filters, 3),
#         #                                      nn.Dropout2d(0.5),
#         #                                      nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))
#
#         self.logits_final = nn.Sequential(ConvRelu(num_filters * 4 + 64, num_filters, 3),
#                                           nn.Dropout2d(0.5),
#                                           nn.Conv2d(num_filters, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#         conv1 = self.conv1(x)     # 1/2
#         conv2 = self.conv2(conv1) # 1/2
#         conv3 = self.conv3(conv2) # 1/4
#         conv4 = self.conv4(conv3) # 1/8
#         conv5 = self.conv5(conv4) # 1/16
#
#         center_2048 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_2048)
#         # ori
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         center = self.center(self.center_se(self.pool(conv5))) # 1/16
#
#         dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1))) # 1/8
#         dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))  # 1/4
#         dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))  # 1/2
#         dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))  # 1
#
#         dec0_add_center = torch.cat((
#             dec2,
#             F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
#         x_final = self.logits_final(dec0_add_center)
#
#         x_dsn = self.dsn(dec4)
#         x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final
#
# class SaltNetSRX101(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None,  **kwargs):
#         assert backbone == 'se_resnext101_32x4d'
#         super(SaltNetSRX101, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         self.nclass = nclass
#
#         num_filters = 32
#         self.encoder = self.pretrained
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
#                                    self.encoder.layer0.bn1,
#                                    self.encoder.layer0.relu1)
#
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
#         self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
#         # ori
#         # self.center_fc = nn.Linear(64, 2)
#
#         self.center_se = SELayer(512*4)
#         self.center = ImprovedIBNaDecoderBlock(512*4,  num_filters * 8)
#
#         self.dec5_se = SELayer(512*4 + num_filters * 8)
#         self.dec5 = ImprovedIBNaDecoderBlock(512*4 + num_filters * 8, num_filters * 8)
#
#         self.dec4_se = SELayer(256*4 + num_filters * 8)
#         self.dec4 = ImprovedIBNaDecoderBlock(256*4 + num_filters * 8, num_filters * 8)
#
#         self.dec3_se = SELayer(128*4 + num_filters * 8)
#         self.dec3 = ImprovedIBNaDecoderBlock(128*4 + num_filters * 8, num_filters * 4)
#
#         self.dec2_se = SELayer(64*4 + num_filters * 4)
#         self.dec2 = ImprovedIBNaDecoderBlock(64*4 + num_filters * 4, num_filters * 4)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(num_filters * 8, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # ori
#         # self.logits_no_empty = nn.Sequential(ConvRelu(num_filters * 4, num_filters, 3),
#         #                                      nn.Dropout2d(0.5),
#         #                                      nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))
#
#
#         self.logits_final = nn.Sequential(ConvRelu(num_filters * 4, num_filters, 3),
#                                           nn.Dropout2d(0.5),
#                                           nn.Conv2d(num_filters, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#
#         conv1 = self.conv1(x)     # 1/2
#         conv2 = self.conv2(conv1) # 1/2
#         conv3 = self.conv3(conv2) # 1/4
#         conv4 = self.conv4(conv3) # 1/8
#         conv5 = self.conv5(conv4) # 1/16
#
#         center_2048 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_2048)
#         # ori
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         center = self.center(self.center_se(self.pool(conv5))) # 1/16
#
#         dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1))) # 1/8
#         dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))  # 1/4
#         dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))  # 1/2
#         dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))  # 1
#
#         x_final = self.logits_final(dec2)
#
#         x_dsn = self.dsn(dec4)
#         x_dsn = F.upsample(x_dsn, scale_factor=4, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final
#
# class SaltNetSR152(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None, is_hypercol=False, **kwargs):
#         assert backbone == 'se_resnet152'
#         super(SaltNetSR152, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         self.nclass = nclass
#         self.encoder = self.pretrained
#
#         self.is_hypercol = is_hypercol
#
#         self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
#                                    self.encoder.layer0.bn1,
#                                    self.encoder.layer0.relu1)
#
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
#         self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
#         # ori
#         # self.center_fc = nn.Linear(64, 2)
#
#         self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(inplace=True),
#                                     nn.MaxPool2d(kernel_size=2,stride=2))
#
#         self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True))
#         self.decoder5 = DecoderBottleneck(256 + 512, 512, 64)
#
#         self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True))
#         self.decoder4 = DecoderBottleneck(64 + 256, 256, 64)
#
#         self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
#         self.decoder3 = DecoderBottleneck(64 + 128, 128, 64)
#
#         self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
#         self.decoder2 = DecoderBottleneck(64 + 64, 64, 64)
#
#         self.decoder1 = DecoderBottleneck(64, 32, 64)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
#         #                             nn.ReLU(inplace=True),
#         #                             nn.Conv2d(64, 1, kernel_size=1, padding=0))
#
#         if self.is_hypercol:
#             self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#         else:
#             self.logits_final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#         conv1 = self.conv1(x)     # 1/4
#         conv2 = self.conv2(conv1) # 1/4
#         conv3 = self.conv3(conv2) # 1/8
#         conv4 = self.conv4(conv3) # 1/16
#         conv5 = self.conv5(conv4) # 1/32
#
#         center_512 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_512)
#         # ori
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         f = self.center(conv5)
#
#         conv5 = self.dec5_1x1(conv5)
#         d5 = self.decoder5(f, conv5)
#
#         conv4 = self.dec4_1x1(conv4)
#         d4 = self.decoder4(d5, conv4)
#
#         conv3 = self.dec3_1x1(conv3)
#         d3 = self.decoder3(d4, conv3)
#
#         conv2 = self.dec2_1x1(conv2)
#         d2 = self.decoder2(d3, conv2)
#
#         d1 = self.decoder1(d2)
#
#         if self.is_hypercol:
#             hypercol = torch.cat((
#                 d1,
#                 F.upsample(d2, scale_factor=2, mode='bilinear'),
#                 F.upsample(d3, scale_factor=4, mode='bilinear'),
#                 F.upsample(d4, scale_factor=8, mode='bilinear'),
#                 F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
#
#             hypercol = F.dropout2d(hypercol, p=0.50)
#             hypercol_add_center = torch.cat((
#                 hypercol,
#                 F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
#             x_final = self.logits_final(hypercol_add_center)
#         else:
#             x_final = self.logits_final(d1)
#
#         x_dsn = self.dsn(d4)
#         x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final
#
# class SaltNetSE154(BaseNet):
#     def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, num_classes=None, is_hypercol=False, **kwargs):
#         assert backbone == 'senet154'
#         super(SaltNetSE154, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#
#         self.nclass = nclass
#         self.encoder = self.pretrained
#
#         self.is_hypercol = is_hypercol
#
#         self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
#                                    self.encoder.layer0.bn1,
#                                    self.encoder.layer0.relu1,
#                                    self.encoder.layer0.conv2,
#                                    self.encoder.layer0.bn2,
#                                    self.encoder.layer0.relu2,
#                                    self.encoder.layer0.conv3,
#                                    self.encoder.layer0.bn3,
#                                    self.encoder.layer0.relu3
#                                    )
#
#         self.conv2 = self.encoder.layer1
#         self.conv3 = self.encoder.layer2
#         self.conv4 = self.encoder.layer3
#         self.conv5 = self.encoder.layer4
#
#         self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
#         self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
#         self.center_fc = nn.Linear(64, 2)
#
#         self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1),
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(inplace=True),
#                                     nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),
#                                       nn.ReLU(inplace=True))
#         self.decoder5 = DecoderBottleneck(256 + 512, 512, 64)
#
#         self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),
#                                       nn.ReLU(inplace=True))
#         self.decoder4 = DecoderBottleneck(64 + 256, 256, 64)
#
#         self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),
#                                       nn.ReLU(inplace=True))
#         self.decoder3 = DecoderBottleneck(64 + 128, 128, 64)
#
#         self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.decoder2 = DecoderBottleneck(64 + 64, 64, 64)
#
#         self.decoder1 = DecoderBottleneck(64, 32, 64)
#
#         self.dsn = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(64, nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         )
#
#         # ori
#         # self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
#         #                                      nn.ReLU(inplace=True),
#         #                                      nn.Conv2d(64, 1, kernel_size=1, padding=0))
#
#         if self.is_hypercol:
#             self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#         else:
#             self.logits_final = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                                               nn.ReLU(inplace=True),
#                                               nn.Conv2d(64, self.nclass, kernel_size=1, padding=0))
#
#     def forward(self, x):
#         conv1 = self.conv1(x)  # 1/4
#         conv2 = self.conv2(conv1)  # 1/4
#         conv3 = self.conv3(conv2)  # 1/8
#         conv4 = self.conv4(conv3)  # 1/16
#         conv5 = self.conv5(conv4)  # 1/32
#
#         center_512 = self.center_global_pool(conv5)
#         center_64 = self.center_conv1x1(center_512)
#         # ori
#         # center_64_flatten = center_64.view(center_64.size(0), -1)
#         # center_fc = self.center_fc(center_64_flatten)
#
#         f = self.center(conv5)
#
#         conv5 = self.dec5_1x1(conv5)
#         d5 = self.decoder5(f, conv5)
#
#         conv4 = self.dec4_1x1(conv4)
#         d4 = self.decoder4(d5, conv4)
#
#         conv3 = self.dec3_1x1(conv3)
#         d3 = self.decoder3(d4, conv3)
#
#         conv2 = self.dec2_1x1(conv2)
#         d2 = self.decoder2(d3, conv2)
#
#         d1 = self.decoder1(d2)
#
#         if self.is_hypercol:
#             hypercol = torch.cat((
#                 d1,
#                 F.upsample(d2, scale_factor=2, mode='bilinear'),
#                 F.upsample(d3, scale_factor=4, mode='bilinear'),
#                 F.upsample(d4, scale_factor=8, mode='bilinear'),
#                 F.upsample(d5, scale_factor=16, mode='bilinear')), 1)
#             hypercol = F.dropout2d(hypercol, p=0.50)
#             hypercol_add_center = torch.cat((
#                 hypercol,
#                 F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)
#             x_final = self.logits_final(hypercol_add_center)
#         else:
#             x_final = self.logits_final(d1)
#
#         x_dsn = self.dsn(d4)
#         x_dsn = F.upsample(x_dsn, scale_factor=8, mode='bilinear')
#
#         if self.training:
#             return tuple([x_dsn, x_final])
#         else:
#             return x_final
        
def get_saltnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    if backbone in ['atrous_resnet50','se_resnext50_32x4d','resnet50','atrous_se_resnext50_32x4d','atrous_se_resnet50']:
        model = SaltNetSRX50(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
        # model = SaltNetSRX50(6, backbone=backbone, root=root, **kwargs)
    elif backbone == 'resnet34':
        model = SaltNet34(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    elif backbone == 'se_resnext101_32x4d':
        model = SaltNetSRX101(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    elif backbone == 'resnext101_ibn_a':
        model = SaltNetRXIBNA101(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    else:
        raise ValueError('unsupport backbone!')

    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model

if __name__ == '__main__':

    model = get_saltnet(backbone='resnet50').cuda()
    img = torch.randn(2,3,256,256).cuda()
    out = model(img)