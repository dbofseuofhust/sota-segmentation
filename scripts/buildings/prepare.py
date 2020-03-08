import os
import os.path as osp
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

# v = 'trainval'
# imginfos = []
# maskinfos = []
# imgs = allnames
# for w in imgs:
#     imginfos.append(os.path.join('leftImg8bit',  w))
#     maskinfos.append(os.path.join('gtFine',  '{}.png'.format(w.split('.')[0])))

# os.makedirs('datasets/buildings',exist_ok=True)

# with open(os.path.join('datasets/buildings',"{}_fine.txt").format(v),'w') as f:
#     for img, mask in zip(imginfos, maskinfos):
#         f.writelines("{}	{}".format(img,mask)+'\n')


root = r'/data/Dataset/buildings/overlapcrop_data/'

rawimgroot = os.path.join(root,'leftImg8bit')
maskimgroot = os.path.join(root,'gtFine')

allnames = os.listdir(rawimgroot)
random.shuffle(allnames)

v = 'trainval'
imginfos = []
maskinfos = []
imgs = allnames
for w in imgs:
    imginfos.append(os.path.join('leftImg8bit',  w))
    maskinfos.append(os.path.join('gtFine',  '{}.png'.format(w.split('.')[0])))

os.makedirs('datasets/buildings',exist_ok=True)

with open(os.path.join('datasets/buildings',"overlap_{}_fine.txt").format(v),'w') as f:
    for img, mask in zip(imginfos, maskinfos):
        f.writelines("{}	{}".format(img,mask)+'\n')

# class
allnames = os.listdir(maskimgroot)
cnt = 0
total = 0
for u in allnames:
    mask = cv2.imread(osp.join(maskimgroot,u))
    # print(np.unique(mask))
    cnt += np.sum(mask[:,:,0])
    total += mask.shape[0]*mask.shape[1]
    print(np.sum(mask[:,:,0])/(mask.shape[0]*mask.shape[1]))
    # import pdb;pdb.set_trace()
print(cnt,total,cnt/total)

