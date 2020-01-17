import cv2
from PIL import Image
from libtiff import TIFF
import tifffile as tif
import numpy as np
import cv2
import os

root = r'/data/Dataset/ead2020_semantic_segmentation/masks_ead2020/'
saveroot = r'/data/Dataset/ead2020_semantic_segmentation/convert_masks_ead2020/'

os.makedirs(saveroot,exist_ok=True)
for u in os.listdir(root):
    subpath = os.path.join(root,u)
    img_tif = tif.imread(subpath)
    mask = np.zeros(img_tif.shape[1:])
    for v in range(img_tif.shape[0]):
        mask[img_tif[v, :, :] > 0] = v+1
    cv2.imwrite(os.path.join(saveroot,'{}.png'.format(u.split('.')[0])), mask)

# mask = cv2.imread(r'C:\Users\admin\PycharmProjects\Crack\DeepBlueAI\ead-segmentation\scripts\EAD2020_semantic_00007.png',0)
# num_class = 6
# print(np.unique(mask))
# for i in range(num_class):
#     mask[mask == i] = i
# print(np.unique(mask))