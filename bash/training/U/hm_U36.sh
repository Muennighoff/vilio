#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20
topk=${1:--1}

# 36 Feats, Seed 129
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 129 --model U \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/uniter-large.pt --num_pos 6 --contrib --exp U36 --topk $topk

python hm.py --seed 129 --model U \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/uniter-large.pt --num_pos 6 --contrib --exp U36 --topk $topk
