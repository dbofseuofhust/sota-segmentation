from tqdm import tqdm
import glob
import cv2
import numpy as np
import os

# img_dir = '/data/Dataset/EDD2020_release-I_2020-01-15/originalImages/'
# fnames = glob.glob(img_dir +'*.jpg')

img_dir = '/data/Dataset/ISBI/MoNuSAC/convert_monusac/images'
fnames = []
for subdir in os.listdir(img_dir):
    for v in os.listdir(os.path.join(img_dir,subdir)):
        fnames.append(os.path.join(img_dir,subdir,v))

R_means = []
G_means = []
B_means = []
R_stds = []
G_stds = []
B_stds = []
with tqdm(total=len(fnames)) as pbar:
    for fname in fnames:
        im = cv2.imread(fname)
        im = cv2.cvtColor(im ,cv2.COLOR_BGR2RGB)

        im_R = im[: ,: ,0 ] /255
        im_G = im[: ,: ,1 ] /255
        im_B = im[: ,: ,2 ] /255
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)

        pbar.update(1)
a = [R_means ,G_means ,B_means]
b = [R_stds ,G_stds ,B_stds]
mean = [0 ,0 ,0]
std = [0 ,0 ,0]
mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])
std[0] = np.mean(b[0])
std[1] = np.mean(b[1])
std[2] = np.mean(b[2])
print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0] ,mean[1] ,mean[2]))
print('数据集的RGB方差为\n[{},{},{}]'.format(std[0] ,std[1] ,std[2]))

# for disease dataset
# 数据集的RGB平均值为 [0.4906180488868793,0.3271382281614091,0.2540256913073398]
# 数据集的RGB方差为 [0.2272532905848313,0.18719978740803653,0.1592990354733096]