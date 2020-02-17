import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio

maskpath = r'/data/db/ead-segmentation/unet/636_mask.mat'
mask = scio.loadmat(maskpath)
# mask = cv2.imread(maskpath)
# print(mask.shape)
# print(np.unique(mask))
# plt.imshow(mask*40)
# plt.show()