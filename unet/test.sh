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

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname danet101  \
#                                    --base-size 768 \
#                                    --crop-size 512 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/inference.py --dataset crack \
#                                           --model danet \
#                                           --resume-dir crack/danet_model/danet50 \
#                                           --base-size 256 \
#                                           --crop-size 256 \
#                                           --workers 1 \
#                                           --backbone resnet50 \
#                                           --multi-grid \
#                                           --multi-dilation 4 8 16 \
#                                           --epochs 240  \
#                                           --batch-size 8 \
#                                           --lr 0.003 \

#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/inference.py --dataset ead \
#                                           --model danet \
#                                           --resume-dir ead/danet_model/exp1-danet50 \
#                                           --base-size 768 \
#                                           --crop-size 768 \
#                                           --workers 1 \
#                                           --backbone resnet50 \
#                                           --multi-grid \
#                                           --multi-dilation 4 8 16 \
#                                           --epochs 240  \
#                                           --batch-size 8 \
#                                           --lr 0.003 \

#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/inference.py --dataset ead \
#                                           --model danet \
#                                           --resume-dir ead/danet_model/exp1-danet50-warmup10-lr002/ \
#                                           --base-size 768 \
#                                           --crop-size 768 \
#                                           --workers 1 \
#                                           --backbone resnet50 \
#                                           --multi-grid \
#                                           --multi-dilation 4 8 16 \
#                                           --epochs 240  \
#                                           --batch-size 8 \
#                                           --lr 0.003

#DATASET=monusac
#MODEL=danet
#CKPT=${DATASET}/${MODEL}_model/exp2-danet101-warmup10-lr002-crop1024-bs8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/inference.py --dataset ${DATASET} \
#                                           --model ${MODEL} \
#                                           --resume-dir ${CKPT}/ \
#                                           --base-size 1024 \
#                                           --crop-size 1024 \
#                                           --workers 1 \
#                                           --backbone resnet101 \
#                                           --multi-grid \
#                                           --multi-dilation 4 8 16 \
#                                           --epochs 240  \
#                                           --batch-size 8 \
#                                           --lr 0.003 \
#                                           --log-root ${DATASET}

#DATASET=monusac
#MODEL=deeplabv3plus
#CKPT=${DATASET}/${MODEL}_model/exp2-deeplabv3plus101-warmup10-lr002-crop1024-bs8
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/inference.py --dataset ${DATASET} \
#                                           --model ${MODEL} \
#                                           --resume-dir ${CKPT}/ \
#                                           --base-size 1024 \
#                                           --crop-size 1024 \
#                                           --workers 1 \
#                                           --backbone atrous_resnet101 \
#                                           --multi-grid \
#                                           --multi-dilation 4 8 16 \
#                                           --epochs 240  \
#                                           --batch-size 8 \
#                                           --lr 0.003 \
#                                           --log-root ${DATASET}

DATASET=monusac
MODEL=saltnet
CKPT=${DATASET}/${MODEL}_model/exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs
CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/inference.py --dataset ${DATASET} \
                                           --model ${MODEL} \
                                           --resume-dir ${CKPT}/ \
                                           --base-size 512 \
                                           --crop-size 768 \
                                           --workers 1 \
                                           --backbone se_resnext50_32x4d \
                                           --multi-grid \
                                           --multi-dilation 4 8 16 \
                                           --epochs 240  \
                                           --batch-size 8 \
                                           --lr 0.003 \
                                           --log-root ${DATASET}