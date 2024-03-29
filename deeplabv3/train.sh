#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../rep_work_dirs/exp6-flym-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.PIXEL_MEAN "([0.09661545, 0.18356957, 0.21322473])" INPUT.PIXEL_STD "([0.13422933, 0.14724616, 0.19259872])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "flym" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train" DATASETS.QUERY_PATH "query" DATASETS.GALLERY_PATH "gallery" \
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')"

RESUME_FROM=cityscapes/deeplabv3_model/deeplabv3101/checkpoint.pth.tar
 CUDA_VISIBLE_DEVICES=0,1,2,3 python deeplabv3/train.py --dataset cityscapes \
                                    --model  deeplabv3 \
                                    --backbone resnet101 \
                                    --checkname deeplabv3101  \
                                    --base-size 1024 \
                                    --crop-size 768 \
                                    --epochs 240 \
                                    --batch-size 8 \
                                    --lr 0.003 \
                                    --workers 0 \
                                    --multi-grid \
                                    --multi-dilation 4 8 16 \
                                    --ohem False \
                                    --resume ${RESUME_FROM}

# CUDA_VISIBLE_DEVICES=0,1,2,3 python deeplabv3/train.py --dataset cityscapes \
#                                    --model  deeplabv3plus \
#                                    --backbone resnet101 \
#                                    --checkname deeplabv3plus101  \
#                                    --base-size 1024 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 0 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --ohem False

