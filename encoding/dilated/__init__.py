"""Dilated ResNet and DenseNet"""
from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152,se_resnet50,se_resnet101,se_resnet152
# from .resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a,resnet152_ibn_a,IBN
from .resnet_ibn_b import resnet50_ibn_b,resnet101_ibn_b,resnet152_ibn_b
from .resnext_ibn_a import resnext50_ibn_a,resnext101_ibn_a,resnext152_ibn_a
from .se_resnet_ibn_a import se_resnet50_ibn_a,se_resnet101_ibn_a,se_resnet152_ibn_a
from .se_resnet_ibn_b import se_resnet50_ibn_b,se_resnet101_ibn_b,se_resnet152_ibn_b
from .senet import se_resnext50_32x4d,se_resnext101_32x4d,senet154
from .resnet_atrous import get_atrous_resnet
from .senet_atrous import get_atrous_senet
from .xception import get_xception

from .resnet import resnet101_ibn_a,resnet50_ibn_a,IBN
