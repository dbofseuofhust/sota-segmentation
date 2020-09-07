from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import numpy as np
import cv2
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('output_path', help='output dir')
    
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    os.makedirs(args.output_path,exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    n_class = 8
    # 类别对应
    matches = [100, 200, 300, 400, 500, 600, 700, 800]

    all_imgs = os.listdir(args.img)
    for v in tqdm(all_imgs):
        img = os.path.join(args.img,v)
        # test a single image
        result = inference_segmentor(model, img)[0]
        seg_img = np.zeros((256, 256), dtype=np.uint16)
        for c in range(n_class):
            seg_img[result[:, :] == c] = c
        seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        save_img = np.zeros((256, 256), dtype=np.uint16)
        for i in range(256):
            for j in range(256):
                save_img[i][j] = matches[int(seg_img[i][j])]
        cv2.imwrite(os.path.join(args.output_path, v.split('.')[0]+".png"), save_img)

    # mask = cv2.imread('work_dirs/deeplabv3plus_r50-d8_256x256_40k_naicseg/sub/100023.png', cv2.IMREAD_UNCHANGED)
    # print(np.unique(mask))

        # show the results
        # show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
