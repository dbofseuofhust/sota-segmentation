import os,cv2,random
import numpy as np
from PIL import Image
from skimage import io
Image.MAX_IMAGE_PIXELS = None

names = [1, 2, 10, 11, 20, 21]

for x in names:
    images_path = '/data/deeplearning/JingWei/jingwei_round2_train_20190726/image_{}.png'.format(x)
    labels_path = '/data/deeplearning/JingWei/jingwei_round2_train_20190726/image_{}_label.png'.format(x)

    output_data = '/data/deeplearning/JingWei/tianchu_xianyu_seg/train_data_512_{}'.format(x)
    if not os.path.isdir(output_data):
        os.makedirs(output_data)

    output_label = '/data/deeplearning/JingWei/tianchu_xianyu_seg/train_label_512_{}'.format(x)
    if not os.path.isdir(output_label):
        os.makedirs(output_label)

    image_img = io.imread(images_path)
    label_img = io.imread(labels_path)
    count = 0
    for cy in range(0, image_img.shape[0], 256):
        for cx in range(0, image_img.shape[1], 256):
            if image_img[cy, cx, 3] != 0:
                count += 1
                x1 = cx - 256
                y1 = cy - 256
                x3 = cx + 256
                y3 = cy + 256
                if x1 < 0:
                    x1 = 0
                    x3 = 512
                if x3 > image_img.shape[1]:
                    x3 = image_img.shape[1]
                    x1 = image_img.shape[1] - 512
                if y1 < 0:
                    y1 = 0
                    y3 = 512
                if y3 > image_img.shape[0]:
                    y3 = image_img.shape[0]
                    y1 = image_img.shape[0] - 512

                crop_image = image_img[y1:y3, x1:x3]
                crop_label = label_img[y1:y3, x1:x3]
                cv2.imwrite(os.path.join(output_data, str(count) + '.png'), crop_image)
                print(os.path.join(output_data, str(count) + '.png'))
                io.imsave(os.path.join(output_label, str(count) + '.png'), crop_label)

