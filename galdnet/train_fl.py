###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationFocalLosses,SegmentationOHEMLosses
from encoding.nn import SegmentationMultiLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
# from encoding.nn.loss import CriterionDSN, CriterionOhemDSN
# from encoding.models import DataParallelModel, DataParallelCriterion
# from torch.nn import DataParallel
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model


from option import Options


torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        self.logger = utils.create_logger(args.log_root, args.log_name)
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        # data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
        #                'crop_size': args.crop_size, 'logger': self.logger,
        #                'scale': args.scale,
        #                'root':'/data/Dataset/buildings/overlapcrop_data/'}
        # trainset = get_segmentation_dataset(args.dataset, split='overlap_trainval', mode='train',
        #                                             **data_kwargs)                       
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'logger': self.logger,
                       'scale': args.scale}
        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train',
                                            **data_kwargs)
        # trainset = get_segmentation_dataset(args.dataset, split='trainval', mode='train',
        #                                     **data_kwargs)
        
        testset = get_segmentation_dataset(args.dataset, split='val', mode='val',
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=1,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        if args.is_dilated:
            model = get_segmentation_model(args.model, dataset=args.dataset,
                                           backbone=args.backbone,
                                           aux=args.aux, se_loss=args.se_loss,
                                           norm_layer=nn.BatchNorm2d,
                                           base_size=args.base_size, crop_size=args.crop_size,
                                           multi_grid=args.multi_grid,
                                           multi_dilation=args.multi_dilation, is_dilated=args.is_dilated)
        else:
            model = get_segmentation_model(args.model, dataset=args.dataset,
                                           backbone=args.backbone,
                                           aux=args.aux, se_loss=args.se_loss,
                                           norm_layer=nn.BatchNorm2d,
                                           base_size=args.base_size, crop_size=args.crop_size,
                                           multi_grid=args.multi_grid,
                                           multi_dilation=args.multi_dilation)

        self.logger.info(model)
        # optimizer using different LR
        params_list = [{'params': model.parameters(), 'lr': args.lr},]
        optimizer = torch.optim.SGD(params_list,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

        if args.ohem:
            print('using ohem loss.')
            self.criterion = SegmentationOHEMLosses(se_loss=False, se_weight=0.2,
                                                aux=True, aux_weight=0.4, weight=None, nclass=self.nclass)
        else:
            self.criterion = SegmentationFocalLosses(gamma=args.fl_gamma,se_loss=False, se_weight=0.2,
                                                aux=True, aux_weight=0.4, weight=None, nclass=self.nclass)

        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # finetune from a trained model
        if args.ft:
            args.start_epoch = 0
            checkpoint = torch.load(args.ft_resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
            self.logger.info("=> loaded checkpoint '{}' ".format(args.ft_resume))

        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # lr scheduler
        if args.warmup:
            assert args.warmup_epoch is not None
            assert args.mutil_steps is not None
            assert args.gamma is not None
            assert args.warmup_factor is not None
            assert args.warmup_method is not None
            self.scheduler = utils.WarmupMultiStepLR(self.optimizer, [int(v) for v in args.mutil_steps.split(',')],
                                                     args.gamma, args.warmup_factor, args.warmup_epoch,
                                                     args.warmup_method)
        else:
            self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.epochs, len(self.trainloader), logger=self.logger,
                                                lr_step=args.lr_step)
        self.best_pred = 0.0

        self.warmup = args.warmup

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)

        if self.warmup:
            self.scheduler.step()

        for i, (image, target) in enumerate(tbar):
            if not self.warmup:
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            loss = self.criterion(outputs,target.cuda())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        if self.warmup:
            self.logger.info('\n=>Epoches %i, learning rate = %.4f, previous best = %.4f' % (epoch, self.scheduler.get_lr()[0], self.best_pred))
        else:
            self.logger.info('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # save checkpoint every 10 epoch
            filename = "checkpoint_%s.pth.tar"%(epoch+1)
            is_best = False
            if epoch > 99:
                if not epoch % 5:
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                        }, self.args, is_best, filename)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            # outputs = gather(outputs, 0, dim=0)
            # pred = outputs[0]
            pred = gather(outputs, 0, dim=0)
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')

        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
        self.logger.info('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)