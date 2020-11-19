#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 10
topk=${1:--1}

# Seed 135
python pretrain_bertV.py --seed 135 --taskMaskLM --wordMaskRate 0.15 --train pretrain \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 100 --loadpre ./data/model.pth --topk $topk

python hm.py --seed 135 --model V \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V135 --topk $topk

python hm.py --seed 135 --model V \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V135 --topk $topk