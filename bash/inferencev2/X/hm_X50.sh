#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}


# 50 Feats, Seed 60
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 60 --model X \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadfin $loadfin --exp X60

python hm.py --seed 60 --model X \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadfin $loadfin2 --exp X60