import os
import numpy as np
import cv2
import random
import shutil

random.seed(2019)

split = ['train','val']

root = r'/data/Dataset/ead2020_semantic_segmentation'

prerawroot = os.path.join(root,'ead','leftImg8bit')
premaskroot = os.path.join(root,'ead','gtFine')

rawimgroot = os.path.join(root,'images_ead2020')
maskimgroot = os.path.join(root,'convert_masks_ead2020')

os.makedirs(prerawroot,exist_ok=True)
os.makedirs(premaskroot,exist_ok=True)

allnames = os.listdir(rawimgroot)
random.shuffle(allnames)

for v in allnames[:int(0.85*len(allnames))]:
    os.makedirs(os.path.join(prerawroot,'train'),exist_ok=True)
    os.makedirs(os.path.join(premaskroot,'train'),exist_ok=True)
    shutil.copyfile(os.path.join(rawimgroot,v),os.path.join(prerawroot,'train',v))
    shutil.copyfile(os.path.join(maskimgroot,'{}.png'.format(v.split('.')[0])),os.path.join(premaskroot,'train','{}.png'.format(v.split('.')[0])))

for v in allnames[int(0.85*len(allnames)):]:
    os.makedirs(os.path.join(prerawroot,'val'),exist_ok=True)
    os.makedirs(os.path.join(premaskroot,'val'),exist_ok=True)
    shutil.copyfile(os.path.join(rawimgroot,v),os.path.join(prerawroot,'val',v))
    shutil.copyfile(os.path.join(maskimgroot,'{}.png'.format(v.split('.')[0])),os.path.join(premaskroot,'val','{}.png'.format(v.split('.')[0])))

split = ['train', 'val']
for v in split:
    imginfos = []
    maskinfos = []
    imgs = os.listdir(os.path.join(prerawroot, v))
    for w in imgs:
        imginfos.append(os.path.join('leftImg8bit', v, w))
        maskinfos.append(os.path.join('gtFine', v, '{}.png'.format(w.split('.')[0])))

    os.makedirs('datasets/ead',exist_ok=True)

    with open(os.path.join('datasets/ead',"{}_fine.txt").format(v),'w') as f:
        for img, mask in zip(imginfos, maskinfos):
            f.writelines("{}	{}".format(img,mask)+'\n')