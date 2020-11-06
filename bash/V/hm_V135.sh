#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 10 & midsave to 5
topk=${1:--1}
midsave=${2:-2000}

# Seed 135
python pretrain_bertV.py --seed 135 --taskMaskLM --wordMaskRate 0.15 --train pretrain \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 50 --loadpre ./data/model.pth --topk $topk

python hm.py --seed 135 --model V \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadpre ./data/LAST_BV.pth --swa --midsave $midsave --exp V135 --subtrain --topk $topk

python hm.py --seed 135 --model V \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadpre ./data/LAST_BV.pth --swa --midsave $midsave --exp V135 --subtrain --topk $topk