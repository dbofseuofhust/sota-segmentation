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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_se_resnet50 \
#                                    --checkname exp4-deeplabv3plus_sr50-warmup10-lr002-bsize512-csize768-rs  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_se_resnet50 \
#                                    --checkname debug-exp4-deeplabv3plus_sr50-warmup10-lr002-bsize512-csize768-rs  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize640-csize768-rs-cj  \
#                                    --base-size 640 \
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
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize640-csize960-rs-cj  \
#                                    --base-size 640 \
#                                    --crop-size 960 \
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
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize768-csize1152-rs-cj  \
#                                    --base-size 768 \
#                                    --crop-size 1152 \
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
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize896-csize1344-rs-cj  \
#                                    --base-size 896 \
#                                    --crop-size 1344 \
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
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize1024-rs-cj  \
#                                    --base-size 512 \
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
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize1024-rs-cj  \
#                                    --base-size 512 \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize640-rs-cj  \
#                                    --base-size 512 \
#                                    --crop-size 640 \
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
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj-allrt  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj-allrt-ohem  \
#                                    --base-size 512 \
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
#                                    --log-root ${DATASET} \
#                                    --ohem True

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-msrs-cj-allrt  \
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
#                                    --log-root ${DATASET} \
#                                    --scale

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj002002002-allrt  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize896-rs-cj-allrt  \
#                                    --base-size 512 \
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




#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  galdnet \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-galdnet50-warmup10-lr002-bsize512-csize896-rs-cj-allrt  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize768-rs-cj-allrt  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-nowarmup-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-nowarmup-cos-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --log-root ${DATASET} \
#                                    --lr-scheduler cos

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj-ddp  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone xception \
#                                    --checkname exp4-xception-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp4-deeplabv3plus101-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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


#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp4-deeplabv3plus101-warmup10-lr002-bsize512-csize768-rs  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp4-deeplabv3plus101-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_se_resnet50 \
#                                    --checkname exp4-deeplabv3plus_sr50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet152 \
#                                    --checkname exp4-deeplabv3plus152-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --model  galdnet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-galdnet50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_se_resnet50 \
#                                    --checkname exp4-deeplabv3plus_sr50-warmup10-lr002-bsize512-csize768-rs  \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp6-danet50-warmup10-lr002-bsize512-csize768-rs-cj-allrt  \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp6-danet50-warmup10-lr002-bsize512-csize768-rs-cj2212-allrt  \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp6-danet50-warmup10-lr002-bsize512-csize768-rs-cj  \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp6-danet50-warmup10-lr002-bsize768-csize1024-rs-cj-allrt  \
#                                    --base-size 768 \
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
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp6-danet50-warmup10-lr002-bsize512-csize768-rsreversed-cj-allrt  \
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
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname debug-exp4-danet50-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp4-danet101-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp4-danet50-warmup10-lr002-bsize512-csize896-rs-cj-8gpu  \
#                                    --base-size 512 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-danet50-warmup10-lr002-resize1024-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-danet50-warmup10-lr002-seprs-bsize896-csize768-cj-allrt  \
#                                    --base-size 896 \
#                                    --crop-size 768 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-danet50-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-danet50-warmup10-lr002-seprs-bsize1024-csize869  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  galdnet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-galdnet50-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-ms-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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
#                                    --log-root ${DATASET} \
#                                    --scale

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-mswithval-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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
#                                    --log-root ${DATASET} \
#                                    --scale

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt-mstd  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone atrous_se_resnet101 \
#                                    --checkname exp8-danetsr101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone atrous_se_resnext101_32x4d \
#                                    --checkname exp8-danetsrx101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp8-saltnet-nohycol-atrous_resnet50-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt-aspp-dilated  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model psp \
#                                    --backbone resnet50 \
#                                    --checkname exp8-pspnet50-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt-aspp-dilated  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model asp_oc_dsn \
#                                    --backbone resnet101 \
#                                    --checkname exp8-asp_oc_dsn_resnet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --ohem False

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-default-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-default-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup20-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup True \
#                                    --warmup-epoch 20 \
#                                    --mutil-steps 80,140 \
#                                    --warmup-factor 0.1 \
#                                    --warmup-method linear \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet152 \
#                                    --checkname exp8-danet152-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-danet101-warmup10-lr002-seprs-bsize896-csize768-cj-allrt  \
#                                    --base-size 896 \
#                                    --crop-size 768 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp8-deeplabv3plus50-warmup10-lr002-seprs-bsize896-csize768-cj-allrt  \
#                                    --base-size 896 \
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
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp8-deeplabv3plus101-warmup10-lr002-seprs-bsize896-csize768-cj-allrt  \
#                                    --base-size 896 \
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
#CUDA_VISIBLE_DEVICES=4,5,6,7 python emanet/train.py --dataset ${DATASET} \
#                                    --model  emanet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-emanet50-warmup10-lr002-seprs-bsize896-csize768-cj-allrt  \
#                                    --base-size 896 \
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
#                                    --log-root ${DATASET} \
#                                    --stride 8 \
#                                    --stage-num 3 \
#                                    --em-norm 0.9

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model saltnet \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp8-saltnet-nohycol-atrous_resnet50-warmup10-lr002-seprs-bsize896-csize768-cj-allrt-aspp-dilated  \
#                                    --base-size 896 \
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
#                                    --backbone atrous_resnet101 \
#                                    --checkname Exp4-deeplabv3plus101-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet152 \
#                                    --checkname exp4-deeplabv3plus152-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet152 \
#                                    --checkname exp4-deeplabv3plus152-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname Exp4-deeplabv3plus50-warmup10-lr002-bsize512-csize896-rs-cj  \
#                                    --base-size 512 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet50 \
#                                    --checkname exp7-danet50-warmup10-lr002-bsize512-csize768-rs-cj-allrt  \
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
#                                    --model  deeplabv3plus \
#                                    --backbone atrous_resnet50 \
#                                    --checkname exp6-deeplabv3plus50-warmup10-lr002-bsize512-csize768-rs-cj-allrt  \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone resnet101 \
#                                    --checkname exp8-asp_oc_dsn_resnet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  pyramid_oc_dsn \
#                                    --backbone resnet101 \
#                                    --checkname exp8-pyramid_oc_dsn_resnet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone resnet152 \
#                                    --checkname exp8-asp_oc_dsn_resnet152-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone se_resnet101 \
#                                    --checkname exp8-asp_oc_dsn_se_resnet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone se_resnet152 \
#                                    --checkname exp8-asp_oc_dsn_se_resnet152-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model asp_oc_dsn \
#                                    --backbone resnet101 \
#                                    --checkname exp8-asp_oc_dsn_resnet101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.003 \
#                                    --workers 2 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --ohem False

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone se_resnet101 \
#                                    --checkname exp8-danetsr101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone resnet152_ibn_a \
#                                    --checkname exp8-asp_oc_dsn_resnet152_ibn_a-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101_ibn_a \
#                                    --checkname exp8-danetibn101-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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


#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone resnet101_ibn_a \
#                                    --checkname exp8-asp_oc_dsn_resnet101_ibn_a-warmup10-lr002-seprs-bsize1024-csize869-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 869 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  unet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-scseunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  unet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-unet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-unet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# need to be retrain
#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-scseocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  daheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-daheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  hcscseunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-hcscseunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scsehcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-scsehcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scsedaheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-scsedaheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone resnet152 \
#                                    --checkname exp8-scseocheadunet_resnet152-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone se_resnet101 \
#                                    --checkname exp8-scseocheadunet_se_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone se_resnext101_32x4d \
#                                    --checkname exp8-scseocheadunet_se_resnext101_32x4d-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone resnext101_32x4d \
#                                    --checkname exp8-scseocheadunet_resnext101_32x4d-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone resnext101_64x4d \
#                                    --checkname exp8-scseocheadunet_resnext101_64x4d-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  scseocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-scseocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# need to be retrain
#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet152 \
#                                    --checkname exp8-ocheadunet_resnet152-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone se_resnet101 \
#                                    --checkname exp8-ocheadunet_se_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname reproduce-exp8-ocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone densenet201 \
#                                    --checkname exp8-ocheadunet_densenet201-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet152 \
#                                    --checkname exp8-ocheadunet_resnet152-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101_ibn_a \
#                                    --checkname exp8-ocheadunet_resnet101_ibn_a-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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
#                                    --log-root ${DATASET} \
#                                    --is-dilated True

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone atrous_resnet101 \
#                                    --checkname exp8-ocheadunet_atrous_resnet101_ibn_a-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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
#                                    --log-root ${DATASET} \
#                                    --is-dilated True

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-warmup10-lr0002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.002 \
#                                    --workers 8 \
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
#                                    --model  hcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-poly-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup False \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-poly-lr0002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.002 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup False \
#                                    --log-root ${DATASET}

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  hcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname reproduce-exp8-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-poly-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
#                                    --epochs 240 \
#                                    --batch-size 8 \
#                                    --lr 0.02 \
#                                    --workers 8 \
#                                    --multi-grid \
#                                    --multi-dilation 4 8 16 \
#                                    --warmup False \
#                                    --log-root ${DATASET} \
#                                    --lr-scheduler cos

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  cpn \
#                                    --backbone resnet101 \
#                                    --checkname exp8-cpn101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-ocheadunet_resnet101-refine-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

#DATASET=monusac
#CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet50 \
#                                    --checkname exp8-ocheadunet_resnet50-refine-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  ocheadunet \
#                                     --backbone resnet50 \
#                                     --checkname exp8-ocheadunet_resnet50-refine-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                     --base-size 1024 \
#                                     --crop-size 896 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 8 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --warmup True \
#                                     --warmup-epoch 10 \
#                                     --mutil-steps 80,140 \
#                                     --warmup-factor 0.1 \
#                                     --warmup-method linear \
#                                     --log-root ${DATASET}

# experiments in the same dataset as htc
# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  ocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp9-ocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python unet/train.py --dataset ${DATASET} \
#                                    --model  danet \
#                                    --backbone resnet101 \
#                                    --checkname exp9-danet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ocnet/train.py --dataset ${DATASET} \
#                                    --model  asp_oc_dsn \
#                                    --backbone resnet152 \
#                                    --checkname exp9-asp_oc_dsn_resnet152-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  hcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp9-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  hcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp8-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt-gsblur  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

# DATASET=monusac
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                    --model  hcocheadunet \
#                                    --backbone resnet101 \
#                                    --checkname exp9-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt-affine  \
#                                    --base-size 1024 \
#                                    --crop-size 896 \
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

DATASET=buildings2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python unet/train.py --dataset buildings2 \
#                                     --model psp \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-psp_101-b1024-c1024-ms  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 
                                    # --warmup True \
                                    # --warmup-epoch 10 \
                                    # --mutil-steps 80,140 \
                                    # --warmup-factor 0.1 \
                                    # --warmup-method linear \
                                    # --log-root ${DATASET}

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  hcocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-hcocheadunet_101-b1024-c1024-ms  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --warmup True \
#                                     --warmup-epoch 10 \
#                                     --mutil-steps 80,140 \
#                                     --warmup-factor 0.1 \
#                                     --warmup-method linear \
#                                     --log-root ${DATASET}

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  hcocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-hcocheadunet_101-b1024-c1024-ms-poly  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c1024-ms-poly  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16                                     


# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseunet_101-b1024-c1024-ms-poly  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16                                     
                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume pretrain_models/DANet101.pth.tar.2   


# DATASET=cityscapes
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname cityscapes-scseocheadunet_101-b1024-c768-ms-poly  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16  \
#                                     --resume   cityscapes/scseocheadunet_model/cityscapes-scseocheadunet_101-b1024-c768-ms-poly/model_best.pth.tar


                                  
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train_dice.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02-dice001  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 60 \
#                                     --batch-size 8 \
#                                     --lr 0.002 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --dice-weight 0.01 \
#                                     --ft \
#                                     --ft-resume buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02/model_best.pth.tar   

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train_dice.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02-dice05  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 60 \
#                                     --batch-size 8 \
#                                     --lr 0.002 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --dice-weight 0.5 \
#                                     --ft \
#                                     --ft-resume buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02/model_best.pth.tar   


# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/train_dice.py --dataset buildings2 \
#                                     --model  danet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02-dice  \
#                                     --base-size 1024 \
#                                     --crop-size 768 \
#                                     --epochs 60 \
#                                     --batch-size 8 \
#                                     --lr 0.002 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --dice-weight 1.0 \
#                                     --ft \
#                                     --ft-resume buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02/model_best.pth.tar   

# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train_dice.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c1024-ms-poly-dice001  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 60 \
#                                     --batch-size 8 \
#                                     --lr 0.002 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16  \
#                                     --dice-weight 0.01 \
#                                     --ft \
#                                     --ft-resume buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b1024-c1024-ms-poly/model_best.pth.tar



# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c768-ms-poly-trainval  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16  

# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b2048-c1024-ms-poly  \
#                                     --base-size 2048 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16                                     




# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train_fl.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c1024-ms-poly-fl  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16                                     



# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c1024-ms-poly-ft  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 \
#                                     --ft \
#                                     --ft-resume cityscapes/scseocheadunet_model/cityscapes-scseocheadunet_101-b1024-c768-ms-poly/model_best.pth.tar.2   
# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b1024-c1024-ms-poly-overlap_trainval  \
#                                     --base-size 1024 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16  

# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b2048-c1024-ms-poly-overlap_trainval  \
#                                     --base-size 2048 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16  


# DATASET=buildings2
# CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
#                                     --model  scseocheadunet \
#                                     --backbone resnet101 \
#                                     --checkname buildings2-scseocheadunet_101-b2048-c1024-ms-poly-trainval  \
#                                     --base-size 2048 \
#                                     --crop-size 1024 \
#                                     --epochs 240 \
#                                     --batch-size 8 \
#                                     --lr 0.02 \
#                                     --workers 2 \
#                                     --multi-grid \
#                                     --multi-dilation 4 8 16 
CUDA_VISIBLE_DEVICES=4,5,6,7 python galdnet/train.py --dataset ${DATASET} \
                                    --model  scseocheadunet \
                                    --backbone resnet101 \
                                    --checkname debug  \
                                    --base-size 1024 \
                                    --crop-size 1024 \
                                    --epochs 240 \
                                    --batch-size 8 \
                                    --lr 0.02 \
                                    --workers 2 \
                                    --multi-grid \
                                    --multi-dilation 4 8 16     