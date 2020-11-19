#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20
topk=${1:--1}

# 50 VG feats, Seed 84
cp ./data/hm_vg5050.tsv ./data/HM_img.tsv

python pretrain_bertO.py --seed 84 --taskMaskLM --taskMatched --wordMaskRate 0.15 --train pretrain --tsv --tr bert-large-uncased \
--batchSize 16 --lr 0.25e-5 --epochs 8 --num_features 50 --loadpre ./data/pytorch_model.bin --topk $topk

python hm.py --seed 84 --model O \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadpre ./data/LAST_BO.pth --contrib --exp OV50 --topk $topk

python hm.py --seed 84 --model O \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 50 --loadpre ./data/LAST_BO.pth --contrib --exp OV50 --topk $topk
