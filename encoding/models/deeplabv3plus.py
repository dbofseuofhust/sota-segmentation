import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from ..models import BaseNet

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

def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)

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

class Deeplabv3Plus(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, aspp_outdim=256,output_stride=16,shortcut_dim=48,shortcut_kernel=1,**kwargs):
        super(Deeplabv3Plus, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel, dim_out=aspp_outdim, rate=16//output_stride)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=output_stride//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, shortcut_dim, shortcut_kernel, 1, padding=shortcut_kernel//2,bias=True),
            nn.BatchNorm2d(shortcut_dim),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(aspp_outdim+shortcut_dim, aspp_outdim, 3, 1, padding=1,bias=True),
            nn.BatchNorm2d(aspp_outdim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(aspp_outdim, aspp_outdim, 3, 1, padding=1,bias=True),
            nn.BatchNorm2d(aspp_outdim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(aspp_outdim, nclass, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.backbone = self.pretrained

        if not 'atrous_se' in backbone:
            old_dict = model_zoo.load_url(model_urls[backbone])
            model_dict = self.backbone.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            self.backbone.load_state_dict(model_dict)
            print('loading {} imagenet pretrained weights done!'.format(backbone))

        self.backbone_layers = self.backbone.get_layers()

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        xo = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        x_dsn = self.dsn(layers[-2])

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)

        result = F.interpolate(result, size=size, mode='bilinear', align_corners=True)
        x_dsn = F.interpolate(x_dsn, size=size, mode='bilinear', align_corners=True)
        if self.training:
            return tuple([x_dsn,result])
        else:
            return result

def get_deeplabv3plus(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    layers = {
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = Deeplabv3Plus(nclass=datasets[dataset.lower()].NUM_CLASS, backbone=backbone, aspp_outdim=256,output_stride=16,shortcut_dim=48,shortcut_kernel=1)
    return model

# def resnet50_atrous(pretrained=True, os=16, **kwargs):
#     """Constructs a atrous ResNet-50 model."""
#     model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
#     if pretrained:
#         old_dict = model_zoo.load_url(model_urls['resnet50'])
#         model_dict = model.state_dict()
#         old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
#         model_dict.update(old_dict)
#         model.load_state_dict(model_dict)
#     return model

