import cv2
from PIL import Image
import shutil
from libtiff import TIFF
import tifffile as tif
import numpy as np
import cv2
import os

orirawroot = r'/data/Dataset/EDD2020_release-I_2020-01-15/originalImages'
orimaskroot = r'/data/Dataset/EDD2020_release-I_2020-01-15/masks'

saverawroot = r'/data/Dataset/EDD2020_release-I_2020-01-15/convert_disease/images'
savemaskroot = r'/data/Dataset/EDD2020_release-I_2020-01-15/convert_disease/masks'

os.makedirs(saverawroot,exist_ok=True)
os.makedirs(savemaskroot,exist_ok=True)

# convert original images
for v in os.listdir(orirawroot):
    shutil.copyfile(os.path.join(orirawroot, v), os.path.join(saverawroot, v))

# convert original mask
for v in os.listdir(orirawroot):
    fname = v.split('.')[0]
    class_list = ['BE', 'suspicious', 'HGD', 'cancer', 'polyp']
    for k,v in enumerate(class_list):
        if os.path.exists(os.path.join(orimaskroot,"{}_{}.tif".format(fname,v))):
            maskpath = os.path.join(orimaskroot,"{}_{}.tif".format(fname,v))
            img_tif = tif.imread(maskpath)
            mask = np.zeros(img_tif.shape)
            mask[img_tif > 0] = (k + 1)
    cv2.imwrite(os.path.join(savemaskroot, '{}.png'.format(fname)), mask)

