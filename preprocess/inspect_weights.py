import torch
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
    print(model)

    model_path = 'work_dirs/deeplabv3plus_r50-d8_256x256_40k_naicseg/published_deeplabv3plus_r50-d8_256x256_40k_naicseg-2c2453c6.pth'
    ckpt = torch.load(model_path)
    # print(ckpt.keys())
    # print(ckpt['state_dict'].keys())


if __name__ == '__main__':
    main()
