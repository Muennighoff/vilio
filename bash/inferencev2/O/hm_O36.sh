#!/bin/bash

# Loading finetuned without having to move it to ./data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# 36 Feats, Seed 42
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 42 --model O \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin --exp O36

python hm.py --seed 42 --model O \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin2 --exp O36