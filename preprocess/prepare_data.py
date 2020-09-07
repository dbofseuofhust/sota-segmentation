import cv2
import numpy as np
import os
import random
from tqdm import tqdm
random.seed(2020)

data_root = '/data/deeplearning/naic2020seg/train/train'

# 类别对应
matches = [100, 200, 300, 400, 500, 600, 700, 800]

"""make train/valid split""" 
if 0:
    img_dir = os.path.join(data_root, 'image')
    all_imgs = os.listdir(img_dir)
    random.shuffle(all_imgs)

    train_imgs = all_imgs[:int(0.85*len(all_imgs))]
    val_imgs = all_imgs[int(0.85*len(all_imgs)):]

    with open(os.path.join(data_root,'train.txt'),'w') as f:
        for u in train_imgs:
            f.writelines(u.split('.')[0]+'\n')

    with open(os.path.join(data_root,'valid.txt'),'w') as f:
        for u in val_imgs:
            f.writelines(u.split('.')[0]+'\n')

""" convert labels """
if 1:
    gt_dir = os.path.join(data_root, 'label')
    save_gt_dir = os.path.join(data_root, 'convert_label')
    os.makedirs(save_gt_dir,exist_ok=True)

    all_masks = os.listdir(gt_dir)
    for v in tqdm(all_masks):
        mask = cv2.imread(os.path.join(gt_dir,v), cv2.IMREAD_UNCHANGED)
        for m in matches:
            mask[mask == m] = matches.index(m)
        cv2.imwrite(os.path.join(save_gt_dir, v),mask)
        
