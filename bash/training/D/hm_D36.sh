#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20
topk=${1:--1}

# 36 Feats, Seed 147
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 147 --model D \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/pytorch_model_11.bin --contrib --exp D36 --topk $topk

python hm.py --seed 147 --model D \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/pytorch_model_11.bin --contrib --exp D36 --topk $topk