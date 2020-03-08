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


# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model scseocheadunet \
#                                    --backbone resnet101_ibn_a \
#                                    --checkname exp10-scseocheadunet_resnet101_ibn_a-warmup10-lr002-b1024-c1024-ms-poly  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --log-root ${DATASET}

# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model ocheadunetplus \
#                                    --backbone resnet101 \
#                                    --checkname exp10-ocheadunetplus_resnet101-warmup10-lr002-b1024-c1024-ms  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/buildings/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml




