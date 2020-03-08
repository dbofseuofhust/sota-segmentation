# Base Images
## 从天池基础镜像构建
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
FROM  registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3
# RUN pip install  --default-timeout=100 easydict boto3 opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN sudo apt update
# RUN sudo apt-get install -y -q libfontconfig1 libxrender1 libglib2.0-0 libsm6 libxext6 ucspi-tcp git
RUN pip install   --default-timeout=200 torch==0.4.1 torchvision==0.2.0 numpy tqdm nose Pillow scipy requests pretrainedmodels albumentations tifffile  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install   --default-timeout=200 --upgrade Pillow  -i https://pypi.tuna.tsinghua.edu.cn/simple

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /app

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /app/
RUN sudo chmod -R 777 /app
RUN python setup.py develop -i https://pypi.tuna.tsinghua.edu.cn/simple
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
