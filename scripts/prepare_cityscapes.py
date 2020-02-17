import os
import numpy as np

split = ['train','val']

root = r'/data/deeplearning/cityscapes'
rawimgroot = os.path.join(root,'leftImg8bit')
maskimgroot = os.path.join(root,'gtFine')

for v in split:
    imginfos = []
    maskinfos = []
    subdirs = os.listdir(os.path.join(rawimgroot, v))
    for u in subdirs:
        imgs = os.listdir(os.path.join(rawimgroot, v, u))
        for w in imgs:
            # aachen_000002_000019_leftImg8bit.png, aachen_000000_000019_gtFine_labelTrainIds.png
            imginfos.append(os.path.join('leftImg8bit', v, u, w))
            maskinfos.append(os.path.join('gtFine', v, u, w.replace('leftImg8bit', 'gtFine_labelTrainIds')))
    with open(os.path.join('datasets/cityscapes',"{}_fine.txt").format(v),'w') as f:
        for img, mask in zip(imginfos, maskinfos):
            f.writelines("{}	{}".format(img,mask)+'\n')