#coding:utf-8
import pandas as pd
import numpy as np
import random
import os
import os.path as osp
from PIL import Image,ImageDraw
import glob
import cv2
def train_crop_pair(root_dir,save_dir,crop_size=512):
    images_dir = root_dir+"/leftImg8bit/"
    masks_dir = root_dir+"/gtFine/"
    images_savedir = save_dir+"/leftImg8bit/"
    masks_savedir = save_dir+"/gtFine/"
    os.makedirs(images_savedir,exist_ok=True)
    os.makedirs(masks_savedir,exist_ok=True)
    print(images_dir)
    for img_name in os.listdir(images_dir):
        image_fname = osp.join(images_dir,img_name)
        mask_fname = osp.join(masks_dir,img_name)
        image = cv2.imread(image_fname)
        mask = cv2.imread(mask_fname)

        raw_image_h,raw_image_w,_ = image.shape
        roi_x_num = raw_image_w//crop_size
        roi_y_num = raw_image_h//crop_size
        print(roi_x_num,roi_y_num)
        for y in range(roi_y_num):
            for x in range(roi_x_num):
                crop_image = image[y*crop_size:(y+1)*crop_size,x*crop_size:(x+1)*crop_size,:]
                crop_mask = mask[y*crop_size:(y+1)*crop_size,x*crop_size:(x+1)*crop_size]
                cv2.imwrite(osp.join(images_savedir,'{}_{}_{}_{}-'.format(raw_image_h,raw_image_w,y*crop_size,x*crop_size)+img_name),crop_image)
                cv2.imwrite(osp.join(masks_savedir,'{}_{}_{}_{}-'.format(raw_image_h,raw_image_w,y*crop_size,x*crop_size)+img_name),crop_mask)

def train_overlapcrop_pair(root_dir,save_dir,crop_size=512,stride_size=128):
    images_dir = root_dir+"/leftImg8bit/"
    masks_dir = root_dir+"/gtFine/"
    images_savedir = save_dir+"/leftImg8bit/"
    masks_savedir = save_dir+"/gtFine/"
    os.makedirs(images_savedir,exist_ok=True)
    os.makedirs(masks_savedir,exist_ok=True)
    print(images_dir)
    for img_name in os.listdir(images_dir):
        image_fname = osp.join(images_dir,img_name)
        mask_fname = osp.join(masks_dir,img_name)
        image = cv2.imread(image_fname)
        mask = cv2.imread(mask_fname)

        raw_image_h,raw_image_w,_ = image.shape
        roi_x_num = (raw_image_w-crop_size)//stride_size+1
        roi_y_num = (raw_image_h-crop_size)//stride_size+1
        print(roi_x_num,roi_y_num)
        for y in range(roi_y_num):
            for x in range(roi_x_num):
                crop_image = image[y*stride_size:y*stride_size+crop_size,x*stride_size:x*stride_size+crop_size,:]
                crop_mask = mask[y*stride_size:y*stride_size+crop_size,x*stride_size:x*stride_size+crop_size]
                print(np.sum(crop_mask[:,:,0])/(crop_mask.shape[0]*crop_mask.shape[1]))
                if np.sum(crop_mask[:,:,0])/(crop_mask.shape[0]*crop_mask.shape[1]) < 0.1:
                    continue
                cv2.imwrite(osp.join(images_savedir,'{}_{}_{}_{}-'.format(raw_image_h,raw_image_w,y*crop_size,x*crop_size)+img_name),crop_image)
                cv2.imwrite(osp.join(masks_savedir,'{}_{}_{}_{}-'.format(raw_image_h,raw_image_w,y*crop_size,x*crop_size)+img_name),crop_mask)

if __name__ == "__main__":
    data_dir = '/data/Dataset/buildings/data/'
    # crop_out_dir = '/data/Dataset/buildings/crop_data/'

    # train_crop_pair(data_dir,crop_out_dir,crop_size=1024)
    crop_out_dir = '/data/Dataset/buildings/overlapcrop_data/'
    train_overlapcrop_pair(data_dir,crop_out_dir,crop_size=1024,stride_size=512)
    

