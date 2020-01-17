import os
import numpy as np
import cv2
import random
import shutil

random.seed(2019)

split = ['train','val']

root = r'/data/Dataset/ISBI/MoNuSAC'

prerawroot = os.path.join(root,'monusac','leftImg8bit')
premaskroot = os.path.join(root,'monusac','gtFine')

rawimgroot = os.path.join(root,'convert_monusac/images')
maskimgroot = os.path.join(root,'convert_monusac/masks')

os.makedirs(prerawroot,exist_ok=True)
os.makedirs(premaskroot,exist_ok=True)

allnames = os.listdir(rawimgroot)
random.shuffle(allnames)

for v in allnames[:int(0.85*len(allnames))]:
    os.makedirs(os.path.join(prerawroot,'train',v),exist_ok=True)
    os.makedirs(os.path.join(premaskroot,'train',v),exist_ok=True)
    for u in os.listdir(os.path.join(rawimgroot,v)):
        shutil.copyfile(os.path.join(rawimgroot,v,u),os.path.join(prerawroot,'train',v,u))
        shutil.copyfile(os.path.join(maskimgroot,v,'{}.png'.format(u.split('.')[0])),os.path.join(premaskroot,'train',v,'{}.png'.format(u.split('.')[0])))

for v in allnames[int(0.85*len(allnames)):]:
    os.makedirs(os.path.join(prerawroot,'val',v),exist_ok=True)
    os.makedirs(os.path.join(premaskroot,'val',v),exist_ok=True)
    for u in os.listdir(os.path.join(rawimgroot,v)):
        shutil.copyfile(os.path.join(rawimgroot,v,u),os.path.join(prerawroot,'val',v,u))
        shutil.copyfile(os.path.join(maskimgroot,v,'{}.png'.format(u.split('.')[0])),os.path.join(premaskroot,'val',v,'{}.png'.format(u.split('.')[0])))

split = ['train', 'val']
for v in split:
    imginfos = []
    maskinfos = []
    sds = os.listdir(os.path.join(prerawroot, v))
    for sd in sds:
        imgs = os.listdir(os.path.join(prerawroot,v,sd))
        for w in imgs:
            imginfos.append(os.path.join('leftImg8bit', v, sd, w))
            maskinfos.append(os.path.join('gtFine', v, sd, '{}.png'.format(w.split('.')[0])))

    os.makedirs('datasets/monusac',exist_ok=True)

    with open(os.path.join('datasets/monusac',"{}_fine.txt").format(v),'w') as f:
        for img, mask in zip(imginfos, maskinfos):
            f.writelines("{}	{}".format(img,mask)+'\n')