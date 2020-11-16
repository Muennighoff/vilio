#!/bin/bash

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}

# Seed 45
python hm.py --seed 45 --model V \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadfin $loadfin --exp V45

python hm.py --seed 45 --model V \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --reg \
--num_features 100 --loadfin $loadfin2 --exp V45