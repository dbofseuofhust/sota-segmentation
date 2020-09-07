import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class DecoderBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None):
        super().__init__()

        block = []
        block.append(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        )
        block.append(
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

@HEADS.register_module()
class UHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 encoder_channels=None,
                 decoder_channels=(256, 128, 64, 32, 16),
                 **kwargs):
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(UHead, self).__init__(**kwargs)

        # [3072, 768, 384, 128, 32]
        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], 
                                    out_channels[0], 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg, 
                                    act_cfg=self.act_cfg)

        self.layer2 = DecoderBlock(in_channels[1], 
                                    out_channels[1], 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg, 
                                    act_cfg=self.act_cfg)

        self.layer3 = DecoderBlock(in_channels[2], 
                                    out_channels[2], 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg, 
                                    act_cfg=self.act_cfg)

        self.layer4 = DecoderBlock(in_channels[3], 
                                    out_channels[3], 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg, 
                                    act_cfg=self.act_cfg)

        self.layer5 = DecoderBlock(in_channels[4], 
                                    out_channels[4], 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg, 
                                    act_cfg=self.act_cfg)

        self.final_conv = nn.Conv2d(out_channels[4], self.num_classes, kernel_size=1)

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, inputs):
        """Forward function."""
        # inputs is [x0, x1, x2, x3, x4]
        encoder_head = inputs[-1]
        skips = inputs[:-1][::-1]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]]) 
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])

        output = self.final_conv(x)
        return output
