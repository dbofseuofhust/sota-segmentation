import cv2
from PIL import Image
import shutil
from libtiff import TIFF
import tifffile as tif
import numpy as np
import cv2
import os

orirawroot = r'/data/Dataset/ISBI/MoNuSAC/MoNuSAC_images_and_annotations'
orimaskroot = r'/data/Dataset/ISBI/MoNuSAC/Ann_MoNuSAC'

saverawroot = r'/data/Dataset/ISBI/MoNuSAC/convert_monusac/images'
savemaskroot = r'/data/Dataset/ISBI/MoNuSAC/convert_monusac/masks'

os.makedirs(saverawroot,exist_ok=True)
os.makedirs(savemaskroot,exist_ok=True)

# convert original images
for sub_dir in os.listdir(orirawroot):
    if os.path.join(orirawroot,sub_dir).endswith('json'):
        continue

    savesubroot = os.path.join(saverawroot,sub_dir)
    os.makedirs(savesubroot,exist_ok=True)
    for v in os.listdir(os.path.join(orirawroot,sub_dir)):
        if v.endswith('tif'):
            shutil.copyfile(os.path.join(orirawroot,sub_dir,v),os.path.join(savesubroot,v.replace('tif','jpg')))

# convert original mask
for sub_dir in os.listdir(orimaskroot):
    if os.path.join(orirawroot, sub_dir).endswith('json'):
        continue

    savesubroot = os.path.join(savemaskroot,sub_dir)
    os.makedirs(savesubroot,exist_ok=True)
    for v in os.listdir(os.path.join(orimaskroot,sub_dir)):
        oriimgpath = os.path.join(saverawroot,sub_dir,'{}.jpg'.format(v))
        oriimg = cv2.imread(oriimgpath)
        mask = np.zeros(oriimg.shape[:2])
        for idx,u in enumerate(os.listdir(os.path.join(orimaskroot,sub_dir,v))):
            for k in os.listdir(os.path.join(orimaskroot,sub_dir,v,u)):
                if os.path.exists(os.path.join(orimaskroot,sub_dir,v,u,k)):
                    img_tif = tif.imread(os.path.join(orimaskroot,sub_dir,v,u,k))
                    mask[img_tif>0] = idx+1
        cv2.imwrite(os.path.join(savemaskroot,sub_dir,'{}.png'.format(v)),mask)

