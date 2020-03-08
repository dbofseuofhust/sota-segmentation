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


# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/test.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir /data/yyh/segmentation/dbseg/buildings/danet_model/danet101 \
#                                            --base-size 1024 \
#                                            --crop-size 768 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --eval 
                                        #    --multi-scales

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir /data/yyh/segmentation/dbseg/buildings/danet_model/danet101/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 768 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 
                                        #    --multi-scales
                                        #    --visual
# python scripts/buildings/make_sub.py --img_dir buildings/danet_vis\
#                                     --save_dir buildings/results/
                                        #    --eval                             

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/test.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir /data/yyh/segmentation/dbseg/buildings/danet_model/danet101-b1024-c1024/\
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir /data/yyh/segmentation/dbseg/buildings/danet_model/danet101-b1024-c1024/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 
                                        #    --multi-scales

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir /data/yyh/segmentation/dbseg/buildings/danet_model/danet101-b768-c768/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7648
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c768/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 768 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7656
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c768/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 
# 0.7672
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7784
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7729
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7957
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7784
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7729

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# # 0.7906
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-vf/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7967
# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.7794
# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 


# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --visual 

# 0.7945
# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-cj005/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.8026
# CUDA_VISIBLE_DEVICES=0,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model hcocheadunet \
#                                            --resume-dir buildings2/hcocheadunet_model/buildings2-hcocheadunet_101-b1024-c1024-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.8118
# CUDA_VISIBLE_DEVICES=0,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model hcocheadunet \
#                                            --resume-dir buildings2/hcocheadunet_model/buildings2-hcocheadunet_101-b1024-c1024-ms-poly/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# 0.8118
# CUDA_VISIBLE_DEVICES=0,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model scseunet \
#                                            --resume-dir buildings2/scseunet_model/buildings2-scseunet_101-b1024-c1024-ms-poly/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# # 0.8144
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model scseocheadunet \
#                                            --resume-dir buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b1024-c1024-ms-poly/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval

                                        #    --test-folder /data/Dataset/buildings/buildings_testA/ \
                                        #    --eval 

# 0.7743
# CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model psp \
#                                            --resume-dir buildings2/psp_model/buildings2-psp_101-b1024-c1024-ms/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 


# 0.8088
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model danet \
#                                            --resume-dir buildings2/danet_model/buildings2-danet101-b1024-c1024-ft-ms-rt10-lr02-dice05/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /data/Dataset/buildings/buildings_testA/ \
#                                            --eval 



# 0.8237 -> 0.8292
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/buildings/docker_predict.py --dataset buildings2 \
                                           --model scseocheadunet \
                                           --resume-dir buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b2048-c1024-ms-poly/model_best.pth.tar \
                                           --base-size 2048 \
                                           --crop-size 1024 \
                                           --workers 0 \
                                           --backbone resnet101 \
                                           --multi-grid \
                                           --multi-dilation 4 8 16 \
                                           --epochs 240  \
                                           --batch-size 8 \
                                           --lr 0.02 \
                                           --test-folder /data/Dataset/buildings/buildings_testA/ \
                                           --visual
                                        #    --eval

                                           