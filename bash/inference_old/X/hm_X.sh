#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}


# 50 Feats, Seed 60
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 60 --model X \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadfin $loadfin --exp X60 --subtest

python hm.py --seed 60 --model X \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadfin $loadfin2 --exp X60 --subtest --combine

# 72 Feats, Seed 90
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python hm.py --seed 90 --model X \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadfin $loadfin --exp X90 --subtest

python hm.py --seed 90 --model X \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadfin $loadfin2 --exp X90 --subtest --combine

# 36 Feats, Seed 30
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 30 --model X \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 36 --loadfin $loadfin --exp X30 --subtest

python hm.py --seed 30 --model X \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 36 --loadfin $loadfin2 --exp X30 --subtest --combine

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp X365072