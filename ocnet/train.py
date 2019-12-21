##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## Modified by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import torch.nn.functional as F
import numpy.ma as ma
import os.path as osp
from torch.nn.parallel.scatter_gather import gather
from encoding.datasets import get_ocsegmentation_dataset
# from encoding.datasets import get_segmentation_dataset
from encoding.models import get_ocsegmentation_model
from option import Parameters
import torchvision.transforms as transform
import random
import timeit
import logging
import pdb
from tqdm import tqdm
import encoding.utils as utils
from tensorboardX import SummaryWriter
from encoding.nn import CriterionCrossEntropy,  CriterionDSN, CriterionOhemDSN, CriterionOhemDSN_single
from encoding.models import DataParallelModel, DataParallelCriterion
from eval import *

start = timeit.default_timer()

args = Parameters().parse()

def lr_poly(base_lr, epoch, epochs, power):
    return base_lr*((1-float(epoch)/epochs)**(power))

def adjust_learning_rate(optimizer, epoch):
    lr = lr_poly(args.lr, epoch, args.end_epochs, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def validation(args,model,valloader,optimizer,epoch,best_pred,nclass):
    # Fast test during the training
    def eval_batch(model, image, target):
        outputs = model(image)
        # outputs = gather(outputs, 0, dim=0)
        pred = outputs[0][-1]
        target = target.cuda()
        pred = F.upsample(input=pred, size=(target.size(1), target.size(2)), mode='bilinear', align_corners=True)
        correct, labeled = utils.batch_pix_accuracy(pred.data, target)
        inter, union = utils.batch_intersection_union(pred.data, target, nclass)
        return correct, labeled, inter, union

    is_best = False
    model.eval()
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    tbar = tqdm(valloader, desc='\r')

    for i, (image, target,_,_) in enumerate(tbar):
    # for i, (image, target) in enumerate(tbar):
        with torch.no_grad():
                correct, labeled, inter, union = eval_batch(model, image, target)

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

    # new_pred = (pixAcc + mIoU) / 2
    # if new_pred > best_pred:
    #     is_best = True
    #     best_pred = new_pred
    #     utils.save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.module.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'best_pred': best_pred,
    #     }, args, is_best)
    #     return best_pred

# def validation(args,model,valloader,optimizer,epoch,best_pred,nclass,input_size):
#     # Fast test during the training
#     model.eval()
#     tbar = tqdm(valloader, desc='\r')
#
#     confusion_matrix = np.zeros((nclass,nclass))
#
#     for i, (image, label, size, name) in enumerate(tbar):
#         size = size[0].numpy()
#         with torch.no_grad():
#             output = predict_whole_img_w_label(model, image.numpy(), args.num_classes,
#                                                            args.method, scale=float(args.whole_scale),
#                                                            label=Variable(label.long().cuda()))
#
#         seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
#         m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 255))
#         ma.set_fill_value(m_seg_pred, 20)
#         seg_pred = m_seg_pred
#
#         seg_gt = np.asarray(label.numpy()[:, :size[0], :size[1]], dtype=np.int)
#         ignore_index = seg_gt != 255
#         seg_gt = seg_gt[ignore_index]
#         seg_pred = seg_pred[ignore_index]
#         confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
#
#         pos = confusion_matrix.sum(1)
#         res = confusion_matrix.sum(0)
#         tp = np.diag(confusion_matrix)
#         IoU_array = (tp / np.maximum(1.0, pos + res - tp))
#         mIoU = IoU_array.mean()
#         tbar.set_description('mIoU: %.3f' % (mIoU))

def main():
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_name = '{}_{}'.format(str(args.method),args.network)
    logger = utils.create_logger(args.snapshot_dir, log_name)

    writer = SummaryWriter(args.snapshot_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True

    deeplab = get_ocsegmentation_model("_".join([args.network, args.method]), num_classes=args.num_classes)

    logger.info(deeplab)

    model = DataParallelModel(deeplab)
    model.train()
    model.float()
    model.cuda()

    criterion = CriterionCrossEntropy()
    if "dsn" in args.method:
        if args.ohem:
            if args.ohem_single:
                print('use ohem only for the second prediction map.')
                criterion = CriterionOhemDSN_single(thres=args.ohem_thres, min_kept=args.ohem_keep,
                                                    dsn_weight=float(args.dsn_weight))
            else:
                criterion = CriterionOhemDSN(thres=args.ohem_thres, min_kept=args.ohem_keep,
                                             dsn_weight=float(args.dsn_weight), use_weight=True)
        else:
            criterion = CriterionDSN(dsn_weight=float(args.dsn_weight), use_weight=True)

    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainset = get_ocsegmentation_dataset(args.dataset, root=args.data_dir, split='val', mode='val',
                                                           crop_size=input_size,
                                                           scale=args.random_scale, mirror=args.random_mirror,
                                                           network=args.network)
    valset = get_ocsegmentation_dataset(args.dataset, root=args.data_dir, split='val', mode='val',
                                   crop_size=input_size,scale=False, mirror=False,
                                   network=args.network)

    nclass = trainset.NUM_CLASS

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=1, pin_memory=True)

    valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=1, pin_memory=True)

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()

    best_pred = 0

    for epoch in range(args.start_epochs,args.end_epochs):
        train_loss = 0.0
        model.train()
        tbar = tqdm(trainloader)
        for i_iter, batch in enumerate(tbar):
            sys.stdout.flush()
            images, labels, _, _ = batch
            # images, labels = batch
            images = Variable(images.cuda())
            labels = Variable(labels.long().cuda())
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, epoch)
            if args.fix_lr:
                lr = args.lr

            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i_iter + 1)))

            if i_iter % 100 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
        logger.info('Train loss: %.3f' % (train_loss / (i_iter + 1)))
        validation(args=args, model=model, valloader=valloader, optimizer=optimizer, epoch=epoch,
                           best_pred=best_pred, nclass=nclass)

    # for i_iter, batch in enumerate(trainloader):
    #     sys.stdout.flush()
    #     i_iter += args.start_iters
    #     images, labels, _, _ = batch
    #     images = Variable(images.cuda())
    #     labels = Variable(labels.long().cuda())
    #     optimizer.zero_grad()
    #     lr = adjust_learning_rate(optimizer, i_iter)
    #     if args.fix_lr:
    #         lr = args.lr
    #     print('learning_rate: {}'.format(lr))
    #
    #     if 'gt' in args.method:
    #         preds = model(images, labels)
    #     else:
    #         preds = model(images)
    #     loss = criterion(preds, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if i_iter % 100 == 0:
    #         writer.add_scalar('learning_rate', lr, i_iter)
    #         writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
    #     print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))
    #
    #     if i_iter >= args.num_steps - 1:
    #         print('save model ...')
    #         torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'CS_scenes_' + str(args.num_steps) + '.pth'))
    #         break
    #
    #     if i_iter % args.save_pred_every == 0:
    #         print('taking snapshot ...')
    #         torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'CS_scenes_' + str(i_iter) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')

if __name__ == '__main__':
    main()