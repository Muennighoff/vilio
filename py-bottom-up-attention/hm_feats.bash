#!/bin/bash

# Attr feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr \
--minboxes 36 --maxboxes 36

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr \
--minboxes 50 --maxboxes 50

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr \
--minboxes 72 --maxboxes 72

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr \
--minboxes 10 --maxboxes 100

# VG feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vg \
--minboxes 50 --maxboxes 50

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vg \
--minboxes 10 --maxboxes 100