###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import sys
import numpy as np
import random
import math
import glob
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset
import tifffile as tif
import cv2

class BuildingSegmentation(BaseDataset):
    # BASE_DIR = 'cityscapes'
    NUM_CLASS = 2
    def __init__(self, root='/data/Dataset/buildings/crop_data/', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(BuildingSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        # root = os.path.join(root, self.BASE_DIR)
        print(root)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_cityscapes_pairs(root, split)
        # if split != 'vis':
        if split in  ['train','val','trainval']:
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        if self.images[index].endswith('.tif'):
            img_tif = tif.imread(self.images[index])
            img = Image.fromarray(cv2.cvtColor(img_tif,cv2.COLOR_BGR2RGB) )
            # print(img_tif.shape)
        else:
            img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
    
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask
    def _sync_transform(self, img, mask):
        assert self.base_size >= self.crop_size
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

        # # [todo] colorjitter
        # if random.random() < 0.5:
        #     cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
        #     img = cj(img)

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
    
    def _val_sync_transform(self, img, mask):
        # resize
        # print('img.size',img.size)
        # print('crop_size.size',self.crop_size)

        # ow, oh = self.crop_size, self.crop_size
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        
        # return img, self._mask_transform(mask)

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
    
        
    def _mask_transform(self, mask):
        # print(mask.size)
        # print(np.unique(mask))
        # print(np.array(mask).shape)
        target = np.array(mask)[:,:,0].astype('int32')
        target[target == 255] = -1
        
        # print('target:',target.shape)
        # print('target:',np.unique(target))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

class BuildingSegmentation2(BaseDataset):
    # BASE_DIR = 'cityscapes'
    NUM_CLASS = 2
    def __init__(self, root='/data/Dataset/buildings/crop_data/', split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(BuildingSegmentation2, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        # root = os.path.join(root, self.BASE_DIR)
        print(root)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_cityscapes_pairs(root, split)
        # if split != 'vis':
        if split in  ['train','val','trainval']:
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        if self.images[index].endswith('.tif'):
            img_tif = tif.imread(self.images[index])
            img = Image.fromarray(cv2.cvtColor(img_tif,cv2.COLOR_BGR2RGB) )
            # print(img_tif.shape)
        else:
            img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
    
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)         
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask
    
    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        # short_size = outsize
        short_size = self.base_size

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
        # ablation 1
        # random hflip 
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        # ablation 2
        # random rotate90
        # if random.random() < 0.5:
        #     img = img.transpose(Image.ROTATE_90)
        #     mask = mask.transpose(Image.ROTATE_90)

        # # random rotate180
        # if random.random() < 0.5:
        #     img = img.transpose(Image.ROTATE_180)
        #     mask = mask.transpose(Image.ROTATE_180)

        # # random rotate270
        # if random.random() < 0.5:
        #     img = img.transpose(Image.ROTATE_270)
        #     mask = mask.transpose(Image.ROTATE_270)
        
        # ablation 4
        # if random.random() < 0.5:
        #     # cj = ColorJitter(brightness=0.02,contrast=0.02,saturation=0.02,hue=0.01)
        #     cj = transform.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.01)
        #     img = cj(img)
        crop_size = self.crop_size
        if self.scale:
            short_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))
            # print(short_size)
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

        # ablation 3
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
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

    # def _mask_transform(self, mask):
    #     return torch.from_numpy(np.array(mask)).long()
    
    def _mask_transform(self, mask):
        # print(mask.size)
        # print(np.unique(mask))
        # print(np.array(mask).shape)
        target = np.array(mask)[:,:,0].astype('int32')
        target[target == 255] = -1
        
        # print('target:',target.shape)
        # print('target:',np.unique(target))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

def _get_cityscapes_pairs(folder, split='train'):
    def get_path_pairs(folder,split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join(folder,ll_str[0].rstrip())
                maskpath = os.path.join(folder,ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths
    if split == 'train':
        split_f = os.path.join('scripts/buildings/datasets/buildings', 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join('scripts/buildings/datasets/buildings', 'val_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'trainval':
        split_f = os.path.join('scripts/buildings/datasets/buildings', 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)    
    elif split == 'overlap_trainval':
        split_f = os.path.join('scripts/buildings/datasets/buildings', 'overlap_trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)      
        
    # elif split == 'test':
    #     split_f = os.path.join('scripts/buildings/datasets/buildings', 'test.txt')
    #     img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'vis':
        mask_paths = []
        img_paths = glob.glob(folder+'/*.tif')
    else:
        # split_f = os.path.join('scripts/buildings/datasets/buildings', 'trainval_fine.txt')
        # img_paths, mask_paths = get_path_pairs(folder, split_f)
        mask_paths = []
        img_paths = glob.glob(folder+'/*.tif')
    return img_paths, mask_paths
