

# training
SAVE_ROOT=/data/deeplearning/naic2020seg/work_dirs

# 2020.09.05
# MODEL=deeplabv3plus_r50-d8_256x256_40k_naicseg
# CONFIG_FILE=configs/deeplabv3plus/${MODEL}.py 
# WORK_DIR=${SAVE_ROOT}/${MODEL}
# CUDA_VISIBLE_DEVICES=1 python tools/train.py ${CONFIG_FILE} --work_dir ${WORK_DIR}

# 2020.09.06
# MODEL=deeplabv3plus_r50-d8_256x256_ohem_40k_naicseg
# CONFIG_FILE=configs/deeplabv3plus/${MODEL}.py 
# CUDA_VISIBLE_DEVICES=1 python tools/train.py ${CONFIG_FILE}

# 2020.09.07
# MODEL=unet_r50_256x256_40k_naicseg
# CONFIG_FILE=configs/unet/${MODEL}.py 
# WORK_DIR=${SAVE_ROOT}/${MODEL}
# CUDA_VISIBLE_DEVICES=3 python tools/train.py ${CONFIG_FILE} --work_dir ${WORK_DIR}

# MODEL=unet_r50_d8_256x256_40k_naicseg
# CONFIG_FILE=configs/unet/${MODEL}.py 
# WORK_DIR=${SAVE_ROOT}/${MODEL}
# CUDA_VISIBLE_DEVICES=1 python tools/train.py ${CONFIG_FILE} --work_dir ${WORK_DIR}

# MODEL=unet_r50_256x256_ohem_40k_naicseg
# CONFIG_FILE=configs/unet/${MODEL}.py 
# WORK_DIR=${SAVE_ROOT}/${MODEL}
# CUDA_VISIBLE_DEVICES=1 python tools/train.py ${CONFIG_FILE} --work_dir ${WORK_DIR}

# MODEL=unet_r50_256x256_40k_naicseg
# CONFIG_FILE=configs/unet/${MODEL}.py 
# WORK_DIR=${SAVE_ROOT}/${MODEL}-rerun
# CUDA_VISIBLE_DEVICES=1 python tools/train.py ${CONFIG_FILE} --work_dir ${WORK_DIR}

# inference
# IMG_ROOT=/data/deeplearning/naic2020seg/test/image_A
# SUB=work_dirs/${MODEL}/sub
# CHECKPOINT=work_dirs/${MODEL}/iter_40000.pth
# CUDA_VISIBLE_DEVICES=1 python tools/inference.py ${IMG_ROOT} ${CONFIG_FILE} ${CHECKPOINT} ${SUB}
# CUDA_VISIBLE_DEVICES=0 python preprocess/inspect_weights.py ${IMG_ROOT} ${CONFIG_FILE} ${CHECKPOINT} ${SUB}
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --format-only --sub-dir ${SUB}

# IMG_ROOT=/data/deeplearning/naic2020seg/test/image_A
# SUB=${WORK_DIR}/sub
# CHECKPOINT=${WORK_DIR}/iter_40000.pth
# CUDA_VISIBLE_DEVICES=2 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --format-only --sub-dir ${SUB}

# publish model
# CUDA_VISIBLE_DEVICES=1 python tools/publish_model.py ${CHECKPOINT} work_dirs/${MODEL}/published_${MODEL}.pth
