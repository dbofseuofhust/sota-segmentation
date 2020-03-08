##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import numpy as np
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch import nn

torch_ver = torch.__version__[:3]

__all__ = ['GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling','SegmentationMultiLosses', "CrossEntropy2d", "OhemCrossEntropy2d",'SegmentationOHEMLosses','CeDiceMultiLosses','SegmentationFocalLosses']

class CrossEntropy2d(Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=False):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight   = use_weight
        if self.use_weight:
            self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).cuda()
            print('CrossEntropy2d weights : {}'.format(self.weight))
        else:
            self.weight = None


    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # Variable(torch.randn(2,10)
        if self.use_weight:
            print('target size {}'.format(target.shape))
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            self.weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = None

        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        # torch.FloatTensor([2.87, 13.19, 5.11, 37.98, 35.14, 30.9, 26.23, 40.24, 6.66, 32.07, 21.08, 28.14, 46.01, 10.35, 44.25, 44.9, 44.25, 47.87, 40.39])
        # weight = Variable(torch.FloatTensor([1, 1.49, 1.28, 1.62, 1.62, 1.62, 1.64, 1.62, 1.49, 1.62, 1.43, 1.62, 1.64, 1.43, 1.64, 1.64, 1.64, 1.64, 1.62]), requires_grad=False).cuda()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss

class OhemCrossEntropy2d(Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=False):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        # if use_weight:
        #     print("w/ class balance")
        #     weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        #     self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        # else:
        #     print("w/o class balance")
        #     self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

        print("w/o class balance")
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)

class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationOHEMLosses(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationOHEMLosses, self).__init__(ignore_label=ignore_index, thresh=0.6, min_kept=100000, use_weight=weight)

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationOHEMLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationOHEMLosses, self).forward(pred1, target)
            loss2 = super(SegmentationOHEMLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationOHEMLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationOHEMLosses, self).forward(pred1, target)
            loss2 = super(SegmentationOHEMLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):

        *preds, target = tuple(inputs)

        preds = tuple(preds)
        loss = 0
        for v in preds:
            loss += super(SegmentationMultiLosses, self).forward(v, target)
        return loss

        # pred1, pred2 ,pred3 = tuple(preds)
        # loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        # loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        # loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        # loss = loss1 + loss2 + loss3
        # return loss

# class SegmentationExtraMultiLosses(CrossEntropyLoss):
#     """2D Cross Entropy Loss with Multi-L1oss"""
#     def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
#         super(SegmentationExtraMultiLosses, self).__init__(weight, size_average, ignore_index)
#         self.nclass = nclass
#
#
#     def forward(self, *inputs):
#
#         *preds, target = tuple(inputs)
#         pred1, pred2, pred3, pred4, pred5 = tuple(preds)
#
#         loss1 = super(SegmentationExtraMultiLosses, self).forward(pred1, target)
#         loss2 = super(SegmentationExtraMultiLosses, self).forward(pred2, target)
#         loss3 = super(SegmentationExtraMultiLosses, self).forward(pred3, target)
#         loss4 = super(SegmentationExtraMultiLosses, self).forward(pred4, target)
#         loss5 = super(SegmentationExtraMultiLosses, self).forward(pred5, target)
#         loss = loss1 + loss2 + loss3 + loss4 + loss5
#         return loss


class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)




# https://github.com/dilligencer-zrj/code_zoo/blob/master/Dice_Loss

def make_one_hot(input, num_classes,ignore_index=-1):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes+1
    shape = tuple(shape)
    _input = torch.ones_like(input)*num_classes
    mask = input == ignore_index
    _input = input.masked_scatter_(mask,_input)
    result = torch.zeros(shape)
    result = result.scatter_(1, _input.cpu(), 1)
    return result[:,:num_classes,:,:]
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape {} vs {} do not match'.format(predict.shape,target.shape)
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

# [todo] test
class CeDiceMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, dice_weight = 0.01,weight=None,size_average=True, ignore_index=-1):
        super(CeDiceMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(weight, ignore_index)

    def forward(self, *inputs):

        *preds, target = tuple(inputs)
        # print(target.min())
        onehot_target = make_one_hot(target.unsqueeze(1),self.nclass)
        onehot_target = onehot_target.cuda()
        preds = tuple(preds)
        loss = 0
        for v in preds:
            loss += super(CeDiceMultiLosses, self).forward(v, target)+ self.dice_weight * self.dice_loss(v,onehot_target)
            # loss += self.dice_weight * self.dice_loss(v,onehot_target)

        return loss
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)

class SegmentationFocalLosses(FocalLoss2d):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, gamma=2,se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationFocalLosses, self).__init__(gamma,weight, size_average, ignore_index)

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationFocalLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationFocalLosses, self).forward(pred1, target)
            loss2 = super(SegmentationFocalLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationFocalLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationFocalLosses, self).forward(pred1, target)
            loss2 = super(SegmentationFocalLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

# class SegmentationDiceLosses(CrossEntropyLoss):
#     """2D Cross Entropy Loss with Auxilary Loss"""
#     def __init__(self, dice_weight = 0.01,se_loss=False, se_weight=0.2, nclass=-1,
#                  aux=False, aux_weight=0.2, weight=None,
#                  size_average=True, ignore_index=-1):
#         super(SegmentationDiceLosses, self).__init__(weight, size_average, ignore_index)

#         self.se_loss = se_loss
#         self.aux = aux
#         self.nclass = nclass
#         self.se_weight = se_weight
#         self.aux_weight = aux_weight
#         self.bceloss = BCELoss(weight, size_average) 
#         self.dice_loss = DiceLoss(weight, ignore_index)
#         self.dice_weight = dice_weight

#     def forward(self, *inputs):
#         if not self.se_loss and not self.aux:
#             return super(SegmentationDiceLosses, self).forward(*inputs)
#         elif not self.se_loss:
#             pred1, pred2, target = tuple(inputs)
#             loss1 = super(SegmentationDiceLosses, self).forward(pred1, target)
#             loss2 = super(SegmentationDiceLosses, self).forward(pred2, target)
#             return loss1 + self.aux_weight * loss2
#         elif not self.aux:
#             pred, se_pred, target = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
#             loss1 = super(SegmentationDiceLosses, self).forward(pred, target)
#             loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
#             return loss1 + self.se_weight * loss2
#         else:
#             pred1, se_pred, pred2, target = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
#             loss1 = super(SegmentationDiceLosses, self).forward(pred1, target)
#             loss2 = super(SegmentationDiceLosses, self).forward(pred2, target)
#             loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
#             return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

#     @staticmethod
#     def _get_batch_label_vector(target, nclass):
#         # target is a 3D Variable BxHxW, output is 2D BxnClass
#         batch = target.size(0)
#         tvect = Variable(torch.zeros(batch, nclass))
#         for i in range(batch):
#             hist = torch.histc(target[i].cpu().data.float(), 
#                                bins=nclass, min=0,
#                                max=nclass-1)
#             vect = hist>0
#             tvect[i] = vect
#         return tvect