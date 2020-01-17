###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
import albumentations as albu
import torchvision.transforms as transforms
import cv2

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480,
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')

        self.min_size,self.max_size = 1024,2048

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def cal_new_size(self, im_h, im_w, min_size, max_size):
        if im_h < im_w:
            if im_h < min_size:
                ratio = 1.0 * min_size / im_h
                im_h = min_size
                im_w = round(im_w * ratio)
            elif im_h > max_size:
                ratio = 1.0 * max_size / im_h
                im_h = max_size
                im_w = round(im_w * ratio)
            else:
                ratio = 1.0
        else:
            if im_w < min_size:
                ratio = 1.0 * min_size / im_w
                im_w = min_size
                im_h = round(im_h * ratio)
            elif im_w > max_size:
                ratio = 1.0 * max_size / im_w
                im_w = max_size
                im_h = round(im_h * ratio)
            else:
                ratio = 1.0
        return im_h, im_w, ratio

    # def _val_sync_transform(self, img, mask):
    #     outsize = self.crop_size
    #     short_size = outsize
    #     w, h = img.size
    #     if w > h:
    #         oh = short_size
    #         ow = int(1.0 * w * oh / h)
    #     else:
    #         ow = short_size
    #         oh = int(1.0 * h * ow / w)
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # center crop
    #     # w, h = img.size
    #     # x1 = int(round((w - outsize) / 2.))
    #     # y1 = int(round((h - outsize) / 2.))
    #     # img = img.crop((x1, y1, x1+outsize, y1+outsize))
    #     # mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
    #
    #     # final transform
    #     return img, self._mask_transform(mask)
    #
    # def _sync_transform(self, img, mask):
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #     crop_size = self.crop_size
    #     if self.scale:
    #         short_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))
    #     else:
    #         short_size = self.base_size
    #     w, h = img.size
    #     if h > w:
    #         ow = short_size
    #         oh = int(1.0 * h * ow / w)
    #     else:
    #         oh = short_size
    #         ow = int(1.0 * w * oh / h)
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #     if short_size < crop_size:
    #         padh = crop_size - oh if oh < crop_size else 0
    #         padw = crop_size - ow if ow < crop_size else 0
    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255) # pad 255 for cityscapes and pad 0 for other datasets
    #     w, h = img.size
    #     x1 = random.randint(0, w - crop_size)
    #     y1 = random.randint(0, h - crop_size)
    #     img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    #     mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
    #     return img, self._mask_transform(mask)

    # def _val_sync_transform(self, img, mask):
    #     # ow, oh = self.base_size, self.base_size
    #     # img = img.resize((ow, oh), Image.BILINEAR)
    #     # mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # pad into the size that can be divided by 32
    #     w,h = img.size
    #     new_w,new_h = (w // 32+1)*32,(h // 32+1)*32
    #     padh = new_h-h
    #     padw = new_w - w
    #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #
    #     return img, self._mask_transform(mask)

    # def _sync_transform(self, img, mask):
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #
    #     img = np.asarray(img)
    #     mask = np.asarray(mask)
    #
    #     # for img size < 512, pad it into crop_size
    #     h,w,c = img.shape
    #     if max(h,w) < self.crop_size:
    #         new_w,new_h = self.crop_size,self.crop_size
    #     elif (h <= self.crop_size) and (w >= self.crop_size):
    #         new_w = w+1
    #         new_h = self.crop_size+1
    #     elif (w <= self.crop_size) and (h >= self.crop_size):
    #         new_h = h+1
    #         new_w = self.crop_size+1
    #     else:
    #         new_w,new_h = w,h
    #
    #     oimg = np.zeros((new_h,new_w,c)).astype(np.uint8)
    #     # omask = np.zeros((new_h,new_w)).astype(np.uint8)
    #     omask = 255*np.ones((new_h, new_w)).astype(np.uint8)
    #     oimg[:h,:w] = img[:,:]
    #     omask[:h,:w] = mask[:,:]
    #
    #     oimg = Image.fromarray(oimg)
    #     omask = Image.fromarray(omask)
    #
    #     # random crop
    #     x1 = random.randint(0, new_w - self.crop_size)
    #     y1 = random.randint(0, new_h - self.crop_size)
    #     img = oimg.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
    #     mask = omask.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
    #
    #     # ow, oh = self.base_size, self.base_size
    #     # img = img.resize((ow, oh), Image.BILINEAR)
    #     # mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # random crop
    #     # x1 = random.randint(0, ow - self.crop_size)
    #     # y1 = random.randint(0, oh - self.crop_size)
    #     # img = img.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
    #     # mask = mask.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
    #
    #     return img, self._mask_transform(mask)

    def _val_sync_transform(self, img, mask):

        # to keep the same scale
        w, h = img.size
        scale = self.crop_size / self.base_size
        ow, oh = int(scale * w), int(scale * h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad into the size that can be divided by 32
        new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
        padh = new_h - oh
        padw = new_w - ow
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random hflip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random rotate90
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)

        # [todo] colorjitter

        # for img size < base_size, pad it into crop_size
        w,h = img.size
        padh = self.crop_size - h if h < self.crop_size else 0
        padw = self.crop_size - w if w < self.crop_size else 0

        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        # random crop
        x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
        y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
        img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
        mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))

        # resize
        ow, oh = self.crop_size, self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(batch[0]))))
