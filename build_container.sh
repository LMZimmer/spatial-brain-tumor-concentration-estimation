#!/bin/bash

# build
docker build -t sbtc -f Dockerfile .

# save as tar
docker save -o sbtc.tar sbtc:latest

# remove image
docker rmi sbtc:latest

# load image from tar
#docker load -i sbtc.tar

# test
#docker run -e WANDB_API_KEY=8c1d319a9c86637eec3b30bc590d763452e95cfa --gpus '"device=0"' sbtc

# check running containers
#docker ps -a

# remove specific container
#dpcker rm <hash>

# remove image
#docker rmi sbtc:latest


