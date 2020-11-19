#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20
topk=${1:--1}


# 50 Feats, Seed 49
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 49 --model D \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadpre ./data/pytorch_model_11.bin --contrib --exp D50 --topk $topk

python hm.py --seed 49 --model D \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadpre ./data/pytorch_model_11.bin --contrib --exp D50 --topk $topk

# 72 Feats, Seed 98
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python hm.py --seed 98 --model D \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 72 --loadpre ./data/pytorch_model_11.bin --contrib --exp D72 --topk $topk

python hm.py --seed 98 --model D \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 72 --loadpre ./data/pytorch_model_11.bin --contrib --exp D72 --topk $topk

# 36 Feats, Seed 147
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 147 --model D \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/pytorch_model_11.bin --contrib --exp D36 --topk $topk

python hm.py --seed 147 --model D \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadpre ./data/pytorch_model_11.bin --contrib --exp D36 --topk $topk

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp D365072