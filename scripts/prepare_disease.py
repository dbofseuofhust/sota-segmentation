import os
import numpy as np
import cv2
import random
import shutil

random.seed(2019)

split = ['train','val']

root = r'/data/Dataset/EDD2020_release-I_2020-01-15'

prerawroot = os.path.join(root,'disease','leftImg8bit')
premaskroot = os.path.join(root,'disease','gtFine')

rawimgroot = os.path.join(root,'convert_disease/images')
maskimgroot = os.path.join(root,'convert_disease/masks')

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

    os.makedirs('datasets/disease',exist_ok=True)

    with open(os.path.join('datasets/disease',"{}_fine.txt").format(v),'w') as f:
        for img, mask in zip(imginfos, maskinfos):
            f.writelines("{}	{}".format(img,mask)+'\n')