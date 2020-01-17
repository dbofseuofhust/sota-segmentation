USE_CLASS_BALANCE=True
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0

#--batch-size 8

SNAPSHOT_DIR="./checkpoint/snapshots/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset cityscapes_oc \
                                    --model  base_oc \
                                    --backbone resnet101 \
                                    --ohem ${USE_OHEM} \
                                    --ohem-thres ${OHEMTHRES} --ohem-keep ${OHEMKEEP} --use-weight ${USE_CLASS_BALANCE} \
                                    --snapshot-dir ${SNAPSHOT_DIR} \
                                    --checkname resnet101_base_oc  \
                                    --base-size 1024 \
                                    --crop-size 769 \
                                    --epochs 240 \
                                    --batch-size 2 \
                                    --learning_rate 1e-2 \
                                    --workers 4 \
                                    --gpu 0,1,2,3
