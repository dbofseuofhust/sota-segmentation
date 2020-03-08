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

#  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname danet101  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16


#  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname danet101-b768-c768  \
#                                     --base-size 768 \
#                                     --crop-size 768 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16


#  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/train.py --dataset buildings \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname danet101-b1024-c1024  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16


#  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c768  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --scale # single scale

# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --scale # single scale


# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ms  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --scale \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2


# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2


# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-vf  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2


# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt10  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-cj005  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.003 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2

