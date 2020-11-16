#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# 50 Feats, Seed 49
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 49 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp D50

python hm.py --seed 49 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin2 --exp D50