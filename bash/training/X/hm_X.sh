#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20 & midsave to 5
topk=${1:--1}

# 50 Feats, Seed 60
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python pretrain_bertX.py --seed 60 --taskMaskLM --wordMaskRate 0.15 --train pretrain --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 50 --loadpre ./data/Epoch18_LXRT.pth --topk $topk

python hm.py --seed 60 --model X \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadpre ./data/LAST_BX.pth --swa --exp X60 --topk $topk

python hm.py --seed 60 --model X \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 50 --loadpre ./data/LAST_BX.pth --swa --exp X60 --topk $topk

# 72 Feats, Seed 90
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python pretrain_bertX.py --seed 90 --taskMaskLM --wordMaskRate 0.15 --train pretrain --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 72 --loadpre ./data/Epoch18_LXRT.pth --topk $topk

python hm.py --seed 90 --model X \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadpre ./data/LAST_BX.pth --swa --exp X90 --topk $topk

python hm.py --seed 90 --model X \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 72 --loadpre ./data/LAST_BX.pth --swa --exp X90 --topk $topk

# 36 Feats, Seed 30
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python pretrain_bertX.py --seed 30 --taskMaskLM --wordMaskRate 0.15 --train pretrain --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--batchSize 16 --lr 0.5e-5 --epochs 8 --num_features 36 --loadpre ./data/Epoch18_LXRT.pth --topk $topk

python hm.py --seed 30 --model X \
--train train --valid dev_seen --test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 36 --loadpre ./data/LAST_BX.pth --swa --exp X30 --topk $topk

python hm.py --seed 30 --model X \
--train traindev --valid dev_seen --test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv --llayers 12 --rlayers 2 --xlayers 5 \
--num_features 36 --loadpre ./data/LAST_BX.pth --swa --exp X30 --topk $topk

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp X365072