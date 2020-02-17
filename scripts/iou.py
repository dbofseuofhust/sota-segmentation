import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os
from tqdm import tqdm

# miou
user = []
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    hist = hist[1:,1:]
    user.append(np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)))
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_mIoU(gt_dir, pred_dir,num_classes):

    imagenames = os.listdir(pred_dir)

    hist = np.zeros((num_classes, num_classes))
    for i,name in enumerate(tqdm(imagenames)):
        gt_img = os.path.join(gt_dir,name[:-6],name)
        pred_img = os.path.join(pred_dir,name)
        pred = np.array(cv2.imread(pred_img)[:,:,0])
        label = np.array(cv2.imread(gt_img)[:, :, 0])
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_img, pred_img))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    mIoUs = per_class_iu(hist)  # 
    print('===+++> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 
    print(mIoUs)
    return mIoUs

num_classes = 5
name_classes = ['1','2','3','4']
gt_dir = '/data/Dataset/ISBI/MoNuSAC/monusac/gtFine/val'
pred_dir = '/data/db/ead-segmentation/monusac/danet_vis'

compute_mIoU(gt_dir,pred_dir,num_classes)

