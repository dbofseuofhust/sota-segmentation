# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]

            if config.MODEL.GRID_EVALUATION:
                # num_classes = 2 # default
                num_classes = config.DATASET.NUM_CLASSES
                # grid evaluation
                h, w, c = size
                # crop_size = 1024 # default
                crop_size = config.TEST.BASE_SIZE
                # pad img into the size that can be divided by crop_size
                cell_h, cell_w = int(h/(crop_size/2)+1), int(w/(crop_size/2)+1)
                ch, cw = cell_h*int(crop_size/2), cell_w*int(crop_size/2)
                docker = np.zeros((1,num_classes,ch,cw))
                docker = torch.from_numpy(docker).float()

                padw , padh = cw-w, ch-h
                image = F.pad(image[:,:,:,:], (0, padw, 0, padh), value=0)

                for j in range(cell_h):
                    for i in range(cell_w):
                        center_x, center_y = (i+1)*crop_size/2-crop_size/4, (j+1)*crop_size/2-crop_size/4
                        patch = np.zeros((1,c,crop_size,crop_size))

                        if i == 0:
                            if j == 0:
                                top_left_x = 0
                                top_left_y = 0
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,int(crop_size/4):,int(crop_size/4):] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            elif j == cell_h-1:
                                top_left_x = center_x-crop_size/4
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/4
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:int(3*crop_size/4),int(crop_size/4):] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            else:
                                top_left_x = center_x-crop_size/4
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:,int(crop_size/4):] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                        elif i == cell_w - 1:
                            if j == 0:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/4
                                bottom_right_x = center_x+crop_size/4
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,int(crop_size/4):,:int(3*crop_size/4)] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            elif j == cell_h-1:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/4
                                bottom_right_y = center_y+crop_size/4
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:int(3*crop_size/4),:int(3*crop_size/4)] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            else:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/4
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:,:int(3*crop_size/4)] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                        else:
                            if j == 0:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/4
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,int(crop_size/4):,:] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            elif j == cell_h-1:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/4
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:int(3*crop_size/4),:] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]
                            else:
                                top_left_x = center_x-crop_size/2
                                top_left_y = center_y-crop_size/2
                                bottom_right_x = center_x+crop_size/2
                                bottom_right_y = center_y+crop_size/2
                                top_left_y, bottom_right_y, top_left_x, bottom_right_x = int(top_left_y), int(bottom_right_y), int(top_left_x), int(bottom_right_x)
                                patch[:,:,:,:] = image[:,:,top_left_y:bottom_right_y,top_left_x:bottom_right_x]

                        patch = torch.from_numpy(patch).float()
                        pred = test_dataset.multi_scale_inference(
                            config,
                            model,
                            patch,
                            scales=config.TEST.SCALE_LIST,
                            flip=config.TEST.FLIP_TEST)

                        if pred.size()[-2] != crop_size or pred.size()[-1] != crop_size:
                            pred = F.interpolate(
                                pred, (crop_size, crop_size),
                                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                            )
                        
                        docker[:,:,int(center_y-crop_size/4):int(center_y+crop_size/4),int(center_x-crop_size/4):int(center_x+crop_size/4)] = pred[:,:,int(crop_size/4):int(3*crop_size/4),int(crop_size/4):int(3*crop_size/4)]

                if sv_pred:
                    sv_path = os.path.join(sv_dir, 'test_results')
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                    test_dataset.save_pred(docker[:,:,:h,:w], sv_path, name)
            else:
                pred = test_dataset.multi_scale_inference(
                    config,
                    model,
                    image,
                    scales=config.TEST.SCALE_LIST,
                    flip=config.TEST.FLIP_TEST)

                if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                    pred = F.interpolate(
                        pred, size[-2:],
                        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                    )

                if sv_pred:
                    sv_path = os.path.join(sv_dir, 'test_results')
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                    test_dataset.save_pred(pred, sv_path, name)
