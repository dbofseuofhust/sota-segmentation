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

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname danet101  \
#                                    --base-size 1024 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname danet50  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-danet50  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ead \
#                                    --model  galdnet \
#                                    --backbone resnet50 \
#                                    --checkname galdnet50  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname danet101  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-danet50-warmup10  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ead \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname saltnet_se_resnext50_32x4d_warmup10  \
#                                    --base-size 512 \
#                                    --crop-size 512 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-danet50-warmup10-lr002  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ead \
#                                    --model  saltnet \
#                                    --backbone resnet34 \
#                                    --checkname saltnet_resnet34_warmup10_lr0_02  \
#                                    --base-size 512 \
#                                    --crop-size 512 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ead \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-danet50-warmup10-lr002-debug  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 120 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 40,70 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ead \
#                                    --model  scseunet \
#                                    --backbone resnet50 \
#                                    --checkname scseunet50-warmup10-lr002  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp2-danet101-warmup10-lr002-crop768-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,3,4 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp2-danet101-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp2-saltnet-se_resnext50_32x4d-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 4 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp1-danet101-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 4 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp1-danet101-warmup10-lr002-crop768-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 4 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp2-deeplabv3plus101-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp2-deeplabv3plus101-warmup10-lr002-crop1024-bs8-aug  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp2-deeplabv3plus101-warmup10-lr002-crop768-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp2-deeplabv3plus101-warmup10-lr002-bs1280-crop1024-bs8  \
#                                    --base-size 1280 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp2-saltnet-se_resnext50_32x4d-warmup10-lr001-crop1024-bs4  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 4 \
#                                    --lr 0.01 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp2-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-crop1024-bs8-rs  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-hf  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-hf-r90  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-aspp  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-aspp-gc  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-saltnet-nohycol-atrous_resnet50-warmup10-lr002-bsize512-csize768-rs-aspp-dilated  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone atrous_se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-atrous_se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-aspp-dilated  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone atrous_se_resnet50 \
#                                    --checkname exp4-saltnet-nohycol-atrous_se_resnet50-warmup10-lr002-bsize512-csize768-rs-aspp-dilated  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize768-rs  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

DATASET=monusac
CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
                                    --model  deeplabv3plus \
                                    --backbone atrous_se_resnet50 \
                                    --checkname exp4-deeplabv3plus_sr50-warmup10-lr002-bsize512-csize768-rs  \
                                    --base-size 512 \
                                    --crop-size 768 \
                                    --epochs 240 \
                                    --batch-size 8 \
                                    --lr 0.02 \
                                    --workers 2 \
                                    --multi-grid \
                                    --multi-dilation 4 8 16 \
                                    --warmup True \
                                    --warmup-epoch 10 \
                                    --mutil-steps 80,140 \
                                    --warmup-factor 0.1 \
                                    --warmup-method linear \
                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext50_32x4d-warmup10-lr002-bsize512-csize768-rs-att  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone se_resnext101_32x4d \
#                                    --checkname exp4-saltnet-nohycol-se_resnext101_32x4d-warmup10-lr002-bsize512-csize768-rs  \
#                                    --base-size 512 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext101_32x4d \
#                                    --checkname exp2-saltnet-nohycol-se_resnext101_32x4d-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  saltnet \
#                                    --backbone se_resnext50_32x4d \
#                                    --checkname exp2-saltnet-nohycol-aspp-se_resnext50_32x4d-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 896 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model scseocheadunet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-scseocheadunet50-warmup10-lr002-crop1024-bs8  \
#                                    --base-size 1024 \
#                                    --crop-size 1024 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 4 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model scseocheadunet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-scseocheadunet50-warmup10-lr002-crop768-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 4 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model scseunet \
#                                    --backbone resnet50 \
#                                    --checkname exp1-scseunet50-warmup10-lr002-crop768-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 4 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=ead
#CUDA_VISIBLE_DEVICES=1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp1-deeplabv3plus50-warmup10-lr002-bs8  \
#                                    --base-size 768 \
#                                    --crop-size 768 \
#                                    --epochs 240 \
#                                    --batch-size 6 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 10 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}