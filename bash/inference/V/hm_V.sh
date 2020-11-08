#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}


# Extract lmdb features
python fts_lmdb/lmdb_conversion.py

# Seed 45
python hm.py --seed 45 --model V \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin --exp V45 --subtest

python hm.py --seed 45 --model V \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin2 --exp V45 --subtest --combine

# Seed 90
python hm.py --seed 90 --model V \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin --exp V90 --subtest

python hm.py --seed 90 --model V \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin2 --exp V90 --subtest --combine

# Seed 135
python hm.py --seed 135 --model V \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin --exp V135 --subtest

python hm.py --seed 135 --model V \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--loadfin $loadfin2 --exp V135 --subtest --combine

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp VLMDB