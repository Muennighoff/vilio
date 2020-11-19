#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20
topk=${1:--1}

# Extract lmdb features
python fts_lmdb/lmdb_conversion.py

# Seed 45
python pretrain_bertV.py --seed 45 --taskMaskLM --wordMaskRate 0.15 --train pretrain \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 100 --loadpre ./data/model.pth --topk $topk

python hm.py --seed 45 --model V \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V45 --topk $topk

python hm.py --seed 45 --model V \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V45 --topk $topk

# Seed 90
python pretrain_bertV.py --seed 90 --taskMaskLM --wordMaskRate 0.15 --train pretrain \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 100 --loadpre ./data/model.pth --topk $topk

python hm.py --seed 90 --model V \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V90 --topk $topk

python hm.py --seed 90 --model V \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V90 --topk $topk


# Seed 135
python pretrain_bertV.py --seed 135 --taskMaskLM --wordMaskRate 0.15 --train pretrain \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 100 --loadpre ./data/model.pth --topk $topk

python hm.py --seed 135 --model V \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V135 --topk $topk

python hm.py --seed 135 --model V \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadpre ./data/LAST_BV.pth --swa --exp V135 --topk $topk

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp VLMDB