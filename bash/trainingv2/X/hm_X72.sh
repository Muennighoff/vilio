#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 10 & midsave to 5
topk=${1:--1}
midsave=${2:-2000}

# 72 Feats, Seed 90
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python pretrain_bertX.py --seed 90 --taskMaskLM --wordMaskRate 0.15 --train pretrain --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 72 --loadpre ./data/Epoch18_LXRT.pth --topk $topk

python hm.py --seed 90 --model X \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadpre ./data/LAST_BX.pth --swa --midsave $midsave --exp X90 --topk $topk

python hm.py --seed 90 --model X \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadpre ./data/LAST_BX.pth --swa --midsave $midsave --exp X90 --topk $topk