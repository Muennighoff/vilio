#!/bin/bash

# Loading finetuned without having to move it to ./data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# 50 Feats, Seed 126
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 126 --model O \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp O50 --subtest

python hm.py --seed 126 --model O \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp O50 --subtest --combine

# 50 VG feats, Seed 84
cp ./data/hm_vg5050.tsv ./data/HM_img.tsv

python hm.py --seed 84 --model O \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp OV50 --subtest

python hm.py --seed 84 --model O \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp OV50 --subtest --combine

# 36 Feats, Seed 42
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 42 --model O \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin --exp O36 --subtest

python hm.py --seed 42 --model O \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin --exp O36 --subtest --combine


# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp O365050