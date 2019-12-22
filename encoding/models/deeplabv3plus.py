import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)
        return x

class Deeplabv3Plus(nn.Module):
    def __init__(self, n_class=19, backbone='resnet50', aspp_outdim=256,output_stride=16,shortcut_dim=48,shortcut_kernel=1):
        super(Deeplabv3Plus, self).__init__()
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
        self.cls_conv = nn.Conv2d(aspp_outdim, n_class, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        layers = {
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }
        atrous = {
            'resnet50': [1, 2, 1],
            'resnet101': [2, 2, 2],
            'resnet152': [1, 1, 1],
        }

        self.backbone = ResNet_Atrous(Bottleneck, layers[backbone], atrous=atrous[backbone], os=output_stride)
        old_dict = model_zoo.load_url(model_urls[backbone])
        model_dict = self.backbone.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        self.backbone.load_state_dict(model_dict)

        self.backbone_layers = self.backbone.get_layers()

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0, bias=True)
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
    model = Deeplabv3Plus(n_class=datasets[dataset.lower()].NUM_CLASS, backbone=backbone, aspp_outdim=256,output_stride=16,shortcut_dim=48,shortcut_kernel=1)
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

