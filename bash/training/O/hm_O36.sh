#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 10 & midsave to 5
topk=${1:--1}

# 36 Feats, Seed 42
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python pretrain_bertO.py --seed 42 --taskMaskLM --taskMatched --wordMaskRate 0.15 --train pretrain --tsv --tr bert-large-uncased \
--batchSize 16 --lr 0.25e-5 --epochs 8 --num_features 36 --loadpre ./data/pytorch_model.bin --topk $topk

python hm.py --seed 42 --model O \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/LAST_BO.pth --contrib --exp O36 --topk $topk

python hm.py --seed 42 --model O \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/LAST_BO.pth --contrib --exp O36 --topk $topk