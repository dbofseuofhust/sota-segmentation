echo "running..."
# 0.734
# CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings \
#                                            --model danet \
#                                            --resume-dir buildings/danet_model/danet101/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 768 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.003 \
#                                            --test-folder /tcdata/buildings_testA

# python scripts/buildings/make_sub.py --img_dir buildings/danet_vis \
#                                     --save_dir buildings/results/

# # 1.1 0.7672
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
#                                            --test-folder /tcdata/buildings_testA

# python scripts/buildings/make_sub.py --img_dir buildings2/danet_vis \
#                                     --save_dir buildings2/results/

# 1.2 0.7957
# 1.3 0.7986

# CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings2 \
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
#                                            --test-folder /tcdata/buildings_testA

# python scripts/buildings/make_sub.py --img_dir buildings2/danet_vis \
#                                     --save_dir buildings2/results/

# 1.4 0.8144
# CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings2 \
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
#                                            --test-folder /tcdata/buildings_testA

# # 1.6
# CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model scseocheadunet \
#                                            --resume-dir buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b1024-c768-ms-poly-trainval/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /tcdata/buildings_testA

# python scripts/buildings/make_sub.py --img_dir buildings2/danet_vis \
#                                     --save_dir buildings2/results/


# CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings2 \
#                                            --model scseocheadunet \
#                                            --resume-dir buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b1024-c1024-ms-poly-overlap_trainval/model_best.pth.tar \
#                                            --base-size 1024 \
#                                            --crop-size 1024 \
#                                            --workers 0 \
#                                            --backbone resnet101 \
#                                            --multi-grid \
#                                            --multi-dilation 4 8 16 \
#                                            --epochs 240  \
#                                            --batch-size 8 \
#                                            --lr 0.02 \
#                                            --test-folder /tcdata/buildings_testA

CUDA_VISIBLE_DEVICES=0 python scripts/buildings/docker_predict.py --dataset buildings2 \
                                           --model scseocheadunet \
                                           --resume-dir buildings2/scseocheadunet_model/buildings2-scseocheadunet_101-b2048-c1024-ms-poly-trainval/model_best.pth.tar \
                                           --base-size 2048 \
                                           --crop-size 1024 \
                                           --workers 0 \
                                           --backbone resnet101 \
                                           --multi-grid \
                                           --multi-dilation 4 8 16 \
                                           --epochs 240  \
                                           --batch-size 8 \
                                           --lr 0.02 \
                                           --test-folder /tcdata/buildings_testA


python scripts/buildings/make_sub.py --img_dir buildings2/danet_vis \
                                    --save_dir buildings2/results/