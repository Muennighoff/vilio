#!/bin/bash

# 36 Feats, Seed 147
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 147 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin ./data/LASTtrain.bin --exp D36 --subtest

python hm.py --seed 147 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin ./data/LASTtraindev.bin --exp D36 --subtest --combine