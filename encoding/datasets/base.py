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
import numbers
from torchvision.transforms import functional as F


__all__ = ['BaseDataset', 'test_batchify_fn']

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class torchCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchCompose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

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
    #
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # pad into the size that can be divided by 32
    #     new_w, new_h = (ow // 32 + 1) * 32, (oh // 32 + 1) * 32
    #     padh = new_h - oh
    #     padw = new_w - ow
    #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
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
    #
    #     # random hflip
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    #
    #     # random rotate90
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_90)
    #         mask = mask.transpose(Image.ROTATE_90)
    #
    #     # random rotate180
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_180)
    #         mask = mask.transpose(Image.ROTATE_180)
    #
    #     # random rotate270
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_270)
    #         mask = mask.transpose(Image.ROTATE_270)
    #
    #     # [todo] colorjitter
    #     if random.random() < 0.5:
    #         cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
    #         img = cj(img)
    #
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
    #
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

    # the best performance now
    # def _val_sync_transform(self, img, mask):
    #
    #     # to keep the same scale
    #     w, h = img.size
    #     scale = self.crop_size / self.base_size
    #     ow, oh = int(scale * w), int(scale * h)
    #
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # pad into the size that can be divided by 32
    #     # ow, oh = w, h
    #     new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #     padh = new_h - oh
    #     padw = new_w - ow
    #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #
    #     # center crop
    #     # w, h = img.size
    #     # x1 = int(round((w - self.base_size) / 2.))
    #     # y1 = int(round((h - self.base_size) / 2.))
    #     # img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #     # mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #
    #     return img, self._mask_transform(mask)
    #
    # def _sync_transform(self, img, mask):
    #
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #
    #     # random hflip
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    #
    #     # random rotate90
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_90)
    #         mask = mask.transpose(Image.ROTATE_90)
    #
    #     # random rotate180
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_180)
    #         mask = mask.transpose(Image.ROTATE_180)
    #
    #     # random rotate270
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_270)
    #         mask = mask.transpose(Image.ROTATE_270)
    #
    #     # [todo] colorjitter
    #     if random.random() < 0.5:
    #         cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
    #         img = cj(img)
    #
    #     # for img size < base_size, pad it into crop_size
    #     w, h = img.size
    #     padh = self.crop_size - h if h < self.crop_size else 0
    #     padw = self.crop_size - w if w < self.crop_size else 0
    #
    #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #
    #     # random crop
    #     x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #     y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #     img = img.crop((x1, y1, x1 + self.base_size, y1 + self.base_size))
    #     mask = mask.crop((x1, y1, x1 + self.base_size, y1 + self.base_size))
    #
    #     # resize
    #     ow, oh = self.crop_size, self.crop_size
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     return img, self._mask_transform(mask)

    def _val_sync_transform(self, img, mask):
        assert self.base_size > self.crop_size

        # to keep the same scale
        w, h = img.size
        # scale = self.crop_size / self.base_size
        # ow, oh = int(scale * w), int(scale * h)

        if (min(w,h) > self.base_size) or (min(w,h)<self.base_size and max(w,h)>self.base_size):
            ow, oh = w, h
            new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
            padh = new_h - oh
            padw = new_w - ow
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
        else:
            ow, oh = self.crop_size, self.crop_size
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        # pad into the size that can be divided by 32
        # ow, oh = w, h
        # new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
        # padh = new_h - oh
        # padw = new_w - ow
        # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        # center crop
        # w, h = img.size
        # x1 = int(round((w - self.base_size) / 2.))
        # y1 = int(round((h - self.base_size) / 2.))
        # img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
        # mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))

        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        assert self.base_size > self.crop_size
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

        # random rotate180
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)

        # random rotate270
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)

        # gaussian blur
        # if random.random() < 0.5:
        #     rad = random.randint(0, 5)
        #     img = img.filter(ImageFilter.GaussianBlur(radius=rad))

        # [todo] colorjitter
        if random.random() < 0.5:
            cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
            img = cj(img)

        # for img size < base_size, pad it into crop_size
        w, h = img.size
        # for large img, random crop
        if min(w,h) > self.base_size:
            # random crop
            x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
            y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # for mid img, pad it and random crop
        elif min(w,h)<self.base_size and max(w,h)>self.base_size:
            padh = self.crop_size - h if h < self.crop_size else 0
            padw = self.crop_size - w if w < self.crop_size else 0

            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

            # random crop
            x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
            y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # for small img, resize it into the same size
        else:
            # resize
            ow, oh = self.crop_size, self.crop_size
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    # def _val_sync_transform(self, img, mask):
    #
    #     # to keep the same scale
    #     w, h = img.size
    #     # scale = self.crop_size / self.base_size
    #     # ow, oh = int(scale * w), int(scale * h)
    #     # img = img.resize((ow, oh), Image.BILINEAR)
    #     # mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     if not min(w,h) > self.crop_size:
    #         short_size = self.crop_size
    #         w, h = img.size
    #         if w > h:
    #             oh = short_size
    #             ow = int(1.0 * w * oh / h)
    #         else:
    #             ow = short_size
    #             oh = int(1.0 * h * ow / w)
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)
    #     else:
    #         ow,oh = w,h
    #
    #     # pad into the size that can be divided by 32
    #     new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #     padh = new_h - oh
    #     padw = new_w - ow
    #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #
    #     return img, self._mask_transform(mask)
    #
    # def _sync_transform(self, img, mask):
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #
    #     # random hflip
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    #
    #     # random rotate90
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_90)
    #         mask = mask.transpose(Image.ROTATE_90)
    #
    #     # random rotate180
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_180)
    #         mask = mask.transpose(Image.ROTATE_180)
    #
    #     # random rotate270
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_270)
    #         mask = mask.transpose(Image.ROTATE_270)
    #
    #     # [todo] colorjitter
    #     if random.random() < 0.5:
    #         cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
    #         img = cj(img)
    #
    #     # for img size < base_size, pad it into crop_size
    #     w,h = img.size
    #     # padh = self.crop_size - h if h < self.crop_size else 0
    #     # padw = self.crop_size - w if w < self.crop_size else 0
    #     # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #
    #     # for img size < base_size, resize the shortest side into crop_size
    #     if not min(w,h) > self.crop_size:
    #         short_size = self.crop_size
    #         w, h = img.size
    #         if w > h:
    #             oh = short_size
    #             ow = int(1.0 * w * oh / h)
    #         else:
    #             ow = short_size
    #             oh = int(1.0 * h * ow / w)
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     # random crop
    #     x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #     y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #     img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #     mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #
    #     # resize
    #     # ow, oh = self.crop_size, self.crop_size
    #     # img = img.resize((ow, oh), Image.BILINEAR)
    #     # mask = mask.resize((ow, oh), Image.NEAREST)
    #
    #     return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

class BaseDatasetV2(data.Dataset):
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

    # def _val_sync_transform(self, img, mask):
    #     assert self.base_size > self.crop_size

    #     # to keep the same scale
    #     w, h = img.size
    #     # scale = self.crop_size / self.base_size
    #     # ow, oh = int(scale * w), int(scale * h)

    #     if (min(w,h) > self.base_size) or (min(w,h)<self.base_size and max(w,h)>self.base_size):
    #         ow, oh = w, h
    #         new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #         padh = new_h - oh
    #         padw = new_w - ow
    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #     else:
    #         ow, oh = self.crop_size, self.crop_size
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)

    #     # pad into the size that can be divided by 32
    #     # ow, oh = w, h
    #     # new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #     # padh = new_h - oh
    #     # padw = new_w - ow
    #     # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    #     # center crop
    #     # w, h = img.size
    #     # x1 = int(round((w - self.base_size) / 2.))
    #     # y1 = int(round((h - self.base_size) / 2.))
    #     # img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #     # mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))

    #     return img, self._mask_transform(mask)

    # def _sync_transform(self, img, mask):
    #     assert self.base_size > self.crop_size
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    #     # random hflip
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    #     # random rotate90
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_90)
    #         mask = mask.transpose(Image.ROTATE_90)

    #     # random rotate180
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_180)
    #         mask = mask.transpose(Image.ROTATE_180)

    #     # random rotate270
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_270)
    #         mask = mask.transpose(Image.ROTATE_270)

    #     # [todo] colorjitter
    #     # if random.random() < 0.5:
    #     #     cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
    #     #     img = cj(img)

    #     # for img size < base_size, pad it into crop_size
    #     w, h = img.size
    #     # for large img, random crop
    #     if min(w,h) > self.base_size:
    #         # random crop
    #         x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #         y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #     # for mid img, pad it and random crop
    #     elif min(w,h)<self.base_size and max(w,h)>self.base_size:
    #         padh = self.crop_size - h if h < self.crop_size else 0
    #         padw = self.crop_size - w if w < self.crop_size else 0

    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    #         # random crop
    #         x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #         y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #     # for small img, resize it into the same size
    #     else:
    #         # resize
    #         ow, oh = self.crop_size, self.crop_size
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)

    #     return img, self._mask_transform(mask)

    def _val_sync_transform(self, img, mask):
        # resize
        ow, oh = self.crop_size, self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        assert self.base_size > self.crop_size
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

        # random rotate180
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)

        # random rotate270
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)

        # # [todo] colorjitter
        # if random.random() < 0.5:
        #     cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
        #     img = cj(img)

        # for img size < base_size, pad it into crop_size
        w, h = img.size
        # resize
        ow, oh = self.crop_size, self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

class BaseDatasetV3(data.Dataset):
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

    # def _val_sync_transform(self, img, mask):
    #     assert self.base_size > self.crop_size

    #     # to keep the same scale
    #     w, h = img.size
    #     # scale = self.crop_size / self.base_size
    #     # ow, oh = int(scale * w), int(scale * h)

    #     if (min(w,h) > self.base_size) or (min(w,h)<self.base_size and max(w,h)>self.base_size):
    #         ow, oh = w, h
    #         new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #         padh = new_h - oh
    #         padw = new_w - ow
    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
    #     else:
    #         ow, oh = self.crop_size, self.crop_size
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)

    #     # pad into the size that can be divided by 32
    #     # ow, oh = w, h
    #     # new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
    #     # padh = new_h - oh
    #     # padw = new_w - ow
    #     # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #     # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    #     # center crop
    #     # w, h = img.size
    #     # x1 = int(round((w - self.base_size) / 2.))
    #     # y1 = int(round((h - self.base_size) / 2.))
    #     # img = img.crop((x1, y1, x1+self.base_size, y1+self.base_size))
    #     # mask = mask.crop((x1, y1, x1+self.base_size, y1+self.base_size))

    #     return img, self._mask_transform(mask)

    # def _sync_transform(self, img, mask):
    #     assert self.base_size > self.crop_size
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    #     # random hflip
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    #     # random rotate90
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_90)
    #         mask = mask.transpose(Image.ROTATE_90)

    #     # random rotate180
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_180)
    #         mask = mask.transpose(Image.ROTATE_180)

    #     # random rotate270
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.ROTATE_270)
    #         mask = mask.transpose(Image.ROTATE_270)

    #     # [todo] colorjitter
    #     # if random.random() < 0.5:
    #     #     cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
    #     #     img = cj(img)

    #     # for img size < base_size, pad it into crop_size
    #     w, h = img.size
    #     # for large img, random crop
    #     if min(w,h) > self.base_size:
    #         # random crop
    #         x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #         y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #     # for mid img, pad it and random crop
    #     elif min(w,h)<self.base_size and max(w,h)>self.base_size:
    #         padh = self.crop_size - h if h < self.crop_size else 0
    #         padw = self.crop_size - w if w < self.crop_size else 0

    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    #         # random crop
    #         x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
    #         y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
    #         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
    #     # for small img, resize it into the same size
    #     else:
    #         # resize
    #         ow, oh = self.crop_size, self.crop_size
    #         img = img.resize((ow, oh), Image.BILINEAR)
    #         mask = mask.resize((ow, oh), Image.NEAREST)

    #     return img, self._mask_transform(mask)

    def _val_sync_transform(self, img, mask):
        # resize
        ow, oh = self.crop_size, self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        assert self.base_size > self.crop_size
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

        # random rotate180
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)

        # random rotate270
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)

        # # [todo] colorjitter
        # if random.random() < 0.5:
        #     cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
        #     img = cj(img)

        # for img size < base_size, pad it into crop_size
        w, h = img.size
        # resize
        ow, oh = self.crop_size, self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

class BaseDatasetV4(data.Dataset):
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


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.scale:
            short_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))
        else:
            short_size = self.base_size
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)#pad 255 for cityscapes
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        # final transform
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
