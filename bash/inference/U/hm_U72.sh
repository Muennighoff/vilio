#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# 72 Feats, Seed 86
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python hm.py --seed 86 --model U \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 72 --num_pos 6 --loadfin $loadfin --exp U72

python hm.py --seed 86 --model U \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 72 --num_pos 6 --loadfin $loadfin2 --exp U72

