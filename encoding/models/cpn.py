from .unet import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet
from ..models import BaseNet

class CPN(BaseNet):
    # def __init__(self, resnet, output_shape, num_class, pretrained=True):
    #     super(CPN, self).__init__()
    def __init__(self, nclass, backbone, aux = False, se_loss = False, norm_layer = nn.BatchNorm2d, ** kwargs):
        super(CPN, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        channel_settings = [2048, 1024, 512, 256]
        self.pretrained = eval('torchvision.models.{}'.format(backbone))(True)
        self.global_net = globalNet(channel_settings, nclass)
        self.refine_net = refineNet(channel_settings[-1], nclass)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x1 = self.pretrained.layer1(x)
        x2 = self.pretrained.layer2(x1)
        x3 = self.pretrained.layer3(x2)
        x4 = self.pretrained.layer4(x3)
        res_out = [x4,x3,x2,x1]

        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)
        return tuple([refine_out]+global_outs)

def get_cpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = CPN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    print('loading {} imagenet pretrained weights done!'.format(backbone))
    return model