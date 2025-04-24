#!/bin/bash
docker build -t sbtc -f Dockerfile .
docker save -o sbtc.tar sbtc:latest
docker rmi sbtc:latest

#docker load -i sbtc.tar
