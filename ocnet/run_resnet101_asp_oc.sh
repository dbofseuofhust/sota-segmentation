#USE_CLASS_BALANCE=False
#USE_OHEM=False
#OHEMTHRES=0.7
#OHEMKEEP=0
#MODEL=asp_oc_dsn
#BACKBONE=resnet101
#SNAPSHOT_DIR="./cityscapes/${MODEL}_${BACKBONE}/"
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ocnet/train.py --dataset cityscapes_oc \
#                                    --model ${MODEL} \
#                                    --backbone ${BACKBONE} \
#                                    --ohem ${USE_OHEM} \
#                                    --ohem-thres ${OHEMTHRES} --ohem-keep ${OHEMKEEP} --use-weight ${USE_CLASS_BALANCE} \
#                                    --snapshot_dir ${SNAPSHOT_DIR} \
#                                    --checkname resnet101_asp_oc  \
#                                    --base-size 1024 \
#                                    --crop-size 769 \
#                                    --epochs 240 \
#                                    --batch-size 1 \
#                                    --lr 1e-2 \
#                                    --workers 4 \
#                                    --gpu 0,1,2,3

#network config
NETWORK="resnet101"
METHOD="asp_oc_dsn"
DATASET="cityscapes_train"

#training settings
LEARNING_RATE=1e-2
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=100000
BATCHSIZE=2
INPUT_SIZE='512,512'
USE_CLASS_BALANCE=False
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0
USE_VAL_SET=False
USE_EXTRA_SET=False

DATA_DIR='/data/deeplearning'
DATA_LIST_PATH='datasets/cityscapes_oc/train.lst'
RESTORE_FROM='./pretrained_model/resnet101-imagenet.pth'

# Set the Output path of checkpoints, training log.
SNAPSHOT_DIR="./cityscapes/${METHOD}_${NETWORK}/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u ocnet/train.py --network $NETWORK --method $METHOD --random-mirror --random-scale --gpu 0,1,2,3 --batch-size $BATCHSIZE \
  --snapshot-dir $SNAPSHOT_DIR  --num-steps $MAX_ITERS --ohem $USE_OHEM --data-list $DATA_LIST_PATH --weight-decay $WEIGHT_DECAY \
  --input-size $INPUT_SIZE --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --use-val $USE_VAL_SET --use-weight $USE_CLASS_BALANCE \
  --snapshot-dir $SNAPSHOT_DIR --restore-from $RESTORE_FROM --start-iters $START_ITERS --lr $LEARNING_RATE  --use-parallel \
  --use-extra $USE_EXTRA_SET --dataset $DATASET --data-dir $DATA_DIR