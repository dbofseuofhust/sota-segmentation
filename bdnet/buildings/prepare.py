import os
import numpy as np
import cv2
import random
import shutil

random.seed(2019)

split = ['train','val']

root = r'/data/Dataset/buildings/crop_data/'

rawimgroot = os.path.join(root,'leftImg8bit')
maskimgroot = os.path.join(root,'gtFine')

allnames = os.listdir(rawimgroot)
random.shuffle(allnames)


# split = ['train', 'val']
# for v in split:
#     imginfos = []
#     maskinfos = []
#     if v == 'train':
#         imgs = allnames[:int(0.85*len(allnames))]
#     else:
#         imgs = allnames[int(0.85*len(allnames)):]
#     for w in imgs:
#         imginfos.append(os.path.join('leftImg8bit',  w))
#         maskinfos.append(os.path.join('gtFine',  '{}.png'.format(w.split('.')[0])))

#     os.makedirs('datasets/buildings',exist_ok=True)

#     with open(os.path.join('datasets/buildings',"{}_fine.txt").format(v),'w') as f:
#         for img, mask in zip(imginfos, maskinfos):
#             f.writelines("{}	{}".format(img,mask)+'\n')

v = 'trainval'
imginfos = []
maskinfos = []
imgs = allnames
for w in imgs:
    imginfos.append(os.path.join('leftImg8bit',  w))
    maskinfos.append(os.path.join('gtFine',  '{}.png'.format(w.split('.')[0])))

os.makedirs('datasets/buildings',exist_ok=True)

with open(os.path.join('datasets/buildings',"{}_fine.txt").format(v),'w') as f:
    for img, mask in zip(imginfos, maskinfos):
        f.writelines("{}	{}".format(img,mask)+'\n')

# allnames = os.listdir(maskimgroot)
# for u in allnames:

