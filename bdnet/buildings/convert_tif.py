import cv2
from PIL import Image
# from libtiff import TIFF
import tifffile as tif
import numpy as np
import cv2
import os
import glob
import pdb
# /data/Dataset/buildings/buildings_testA
# root = '/data/Dataset/buildings/buildings_train_20200107/'
# saveroot = '/data/Dataset/buildings/data/'

# os.makedirs(saveroot,exist_ok=True)
# os.makedirs(saveroot+'leftImg8bit/',exist_ok=True)
# os.makedirs(saveroot+'gtFine/',exist_ok=True)
# for u in os.listdir(root):
#     subpath = os.path.join(root,u)
#     img_tif = tif.imread(subpath+'/{}_img_RGB.tif'.format(u))
#     cv2.imwrite(os.path.join(saveroot,'leftImg8bit','{}.png'.format(u.split('.')[0])), img_tif)
#     # mask single channel
#     img_tif = tif.imread(subpath+'/{}_building_footprints.tif'.format(u))
#     cv2.imwrite(os.path.join(saveroot,'gtFine','{}.png'.format(u.split('.')[0])), img_tif)


# 
root = '/data/Dataset/buildings/buildings_testA'
saveroot = '/data/Dataset/buildings/test/'

os.makedirs(saveroot,exist_ok=True)

for u in os.listdir(root):
    subpath = os.path.join(root,u)
    img_tif = tif.imread(subpath)
    
    cv2.imwrite(os.path.join(saveroot,u), img_tif)
    img = cv2.imread(os.path.join(saveroot,u))
    import pdb;pdb.set_trace()
    # mask = np.zeros(img_tif.shape[1:])
    # for v in range(img_tif.shape[0]):
    #     mask[img_tif[v, :, :] > 0] = v+1
 
# print(np.unique(mask))
# for i in range(num_class):
#     mask[mask == i] = i
# print(np.unique(mask))
