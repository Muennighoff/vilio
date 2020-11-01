#!/bin/bash

python hm.py --seed 98 --model D \
--train traincleanex --valid devseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 72 --loadpre ../input/devlbert/pytorch_model_11.bin --contrib --midsave 2000