# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.0 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.0  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.0

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.1 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.1  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.1

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.2 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.2  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.2


# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.3 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.3  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.3

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.4 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.4  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.4

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.5 . 
# # sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.5  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.5

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.6 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.6  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.6

# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.7 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.7  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.7


# sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.8 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.8  sh run.sh
# sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.8

sudo docker build -t registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.9 . 
# sudo nvidia-docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.9  sh run.sh
sudo docker push registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.9

# 
# sudo docker run -v /data/Dataset/buildings:/tcdata registry.cn-shenzhen.aliyuncs.com/deepblueai/buildings/buildings:1.0  sh run.sh


# m2
# sudo docker run -it  --shm-size=8g --name dbseg registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3
# sudo docker start dbseg
# sudo docker container exec -it dbseg /bin/bash

#service docker restart
