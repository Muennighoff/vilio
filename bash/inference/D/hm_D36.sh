#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# 36 Feats, Seed 147
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 147 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin --exp D36

python hm.py --seed 147 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin2 --exp D36