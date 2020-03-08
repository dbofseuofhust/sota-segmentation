import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class resconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.Conv_1x1(x)
        x = self.conv(x)

        return residual + x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNNBlock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
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

        return x * psi

class R2UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, first_layer_numKernel=64):
        super(R2UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNNBlock(ch_in=img_ch, ch_out=first_layer_numKernel, t=t)
        self.RRCNN2 = RRCNNBlock(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel, t=t)
        self.RRCNN3 = RRCNNBlock(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel, t=t)
        self.RRCNN4 = RRCNNBlock(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel, t=t)
        self.RRCNN5 = RRCNNBlock(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel, t=t)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel, t=t)
        self.Up_RCNN5 = RRCNNBlock(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel, t=t)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel, t=t)
        self.Up_RCNN4 = RRCNNBlock(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel, t=t)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel, t=t)
        self.Up_RCNN3 = RRCNNBlock(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel, t=t)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel, t=t)
        self.Up_RCNN2 = RRCNNBlock(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel, t=t)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, first_layer_numKernel=64):
        super(AttUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Conv5 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Att5 = AttentionBlock(F_g=8 * first_layer_numKernel, F_l=8 * first_layer_numKernel,
                                    F_int=4 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Att4 = AttentionBlock(F_g=4 * first_layer_numKernel, F_l=4 * first_layer_numKernel,
                                    F_int=2 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Att3 = AttentionBlock(F_g=2 * first_layer_numKernel, F_l=2 * first_layer_numKernel,
                                    F_int=first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Att2 = AttentionBlock(F_g=first_layer_numKernel, F_l=first_layer_numKernel,
                                    F_int=int(first_layer_numKernel / 2))
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, first_layer_numKernel=64):
        super(R2AttUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNNBlock(ch_in=img_ch, ch_out=first_layer_numKernel, t=t)
        self.RRCNN2 = RRCNNBlock(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel, t=t)
        self.RRCNN3 = RRCNNBlock(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel, t=t)
        self.RRCNN4 = RRCNNBlock(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel, t=t)
        self.RRCNN5 = RRCNNBlock(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel, t=t)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Att5 = AttentionBlock(F_g=8 * first_layer_numKernel, F_l=8 * first_layer_numKernel,
                                    F_int=4 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Att4 = AttentionBlock(F_g=4 * first_layer_numKernel, F_l=4 * first_layer_numKernel,
                                    F_int=2 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Att3 = AttentionBlock(F_g=2 * first_layer_numKernel, F_l=2 * first_layer_numKernel,
                                    F_int=first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Att2 = AttentionBlock(F_g=first_layer_numKernel, F_l=first_layer_numKernel,
                                    F_int=int(first_layer_numKernel / 2))
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ResAttUNet(nn.Module):
    def __init__(self, UnetLayer, img_ch=1, output_ch=1, first_layer_numKernel=64):
        super(ResAttUNet, self).__init__()

        self.UnetLayer = UnetLayer
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = resconv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = resconv_block(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = resconv_block(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = resconv_block(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Conv5 = resconv_block(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Att5 = AttentionBlock(F_g=8 * first_layer_numKernel, F_l=8 * first_layer_numKernel,
                                    F_int=4 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Att4 = AttentionBlock(F_g=4 * first_layer_numKernel, F_l=4 * first_layer_numKernel,
                                    F_int=2 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Att3 = AttentionBlock(F_g=2 * first_layer_numKernel, F_l=2 * first_layer_numKernel,
                                    F_int=first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Att2 = AttentionBlock(F_g=first_layer_numKernel, F_l=first_layer_numKernel,
                                    F_int=int(first_layer_numKernel / 2))
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)

        if self.UnetLayer > 3:
            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)

            d4 = self.Up4(x4)

            if self.UnetLayer > 4:
                x5 = self.Maxpool(x4)
                x5 = self.Conv5(x5)

                # decoding + concat path
                d5 = self.Up5(x5)
                x4 = self.Att5(g=d5, x=x4)
                d5 = torch.cat((x4, d5), dim=1)
                d5 = self.Up_conv5(d5)

                d4 = self.Up4(d5)

            x3 = self.Att4(g=d4, x=x3)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1