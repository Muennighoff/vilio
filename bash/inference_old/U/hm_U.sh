#!/bin/bash

# Del this file?

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}


# 50 Feats, Seed 43
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 43 --model U \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 50 --num_pos 6 --loadfin $loadfin --exp U43 --subtest

python hm.py --seed 43 --model U \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 50 --num_pos 6 --loadfin $loadfin2 --exp U43 --subtest --combine

# 72 Feats, Seed 86
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python hm.py --seed 86 --model U \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 72 --num_pos 6 --loadfin $loadfin --exp U72 --subtest

python hm.py --seed 86 --model U \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 72 --num_pos 6 --loadfin $loadfin2 --exp U72 --subtest --combine

# 36 Feats, Seed 129
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 129 --model U \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 36 --num_pos 6 --loadfin $loadfin --exp U36 --subtest

python hm.py --seed 129 --model U \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-large-cased --epochs 5 --tsv \
--num_features 36 --num_pos 6 --loadfin $loadfin2 --exp U36 --subtest --combine

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp U365072