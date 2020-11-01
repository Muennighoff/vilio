
## Outline to reproduce the ROC-AUC score on Hateful Memes

For the purpose of having everything in one repo, I combined three separate repo's here, which should be treated as separate (in terms of package requirements, as they conflict):
- py-bottom-up (Only for the purpose of feature extraction)
- ernie-vil (For running E-Models)
- vilio (For X, D, V, U, O-Models & everything else)

The pipeline to reproduce the roc-auc score on the public & private leaderboard on the Hateful Memes challenge follows:

### Soft- & Hardware

I used one NVIDIA Tesla P100, Cuda 10.2 & Python 3 for all purposes.
A better GPU/multiple-GPUs will significantly speed up things, but I made sure everything works with just those basics. 
We will install specific packages for each subprocess as outlined below. 


### Data preparation

1. Images
We perform feature extraction before starting to train to speed up training (I also have the code for feature extraction on the go if ever needed (will double training time though)).

Refer to the feature_extraction notebook under notebooks if you run into any problems:

a) Clone the repo;
git clone https://github.com/Muennighoff/vilio.git 

b) Setup extraction:
cd vilio/py-bottom-up-attention; pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd vilio/py-bottom-up-attention; python setup.py build develop

This will take a couple of minutes, if you run into any problems, refer to the README under vilio/py-bottom-up-attention/README.md or the notebook. 

c) Place the img folder of the HM-challenge in vilio/py-bottom-up-attention/data; Make sure to place the whole img folder in there, i.e. the images will be at vilio/py-bottom-up-attention/data/img/

d) Extract features - We use 6 different features for diversity in our model. Run them all at once with:
cd vilio/py-bottom-up-attention; bash hm_feats.bash

Alternatively run them individually with the commands in hm_feats.bash, e.g.:

cd vilio/py-bottom-up-attention; python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr \
--minboxes 36 --maxboxes 36

On the hardware I used, each extraction takes about 90 minutes. 

After completing extraction, we have the following files in vilio/py-bottom-up-attention/data:
- HM_vgattr3636.tsv
- HM_vgattr5050.tsv
- HM_vgattr7272.tsv
- HM_vgattr10100.tsv
- HM_vg5050.tsv
- HM_vg10100.tsv

We also make use of the lmdb feats provided by mmf for one of our models, download them at:
https://dl.fbaipublicfiles.com/mmf/data/datasets/hateful_memes/defaults/features/features.tar.gz

With the downloaded detectron.lmdb, we now have 7 different feature files.

2. Text 
- Download the updated train.jsonl, dev_seen.jsonl, dev_unseen.jsonl, test_seen.jsonl, test_unseen.jsonl from the HM Challenge and place them in BOTH the data folder under vilio/data and the data folder under vilio/ernie-vil/data.


### Individual Model Pretraining & Training

1. PyTorch / U O V D X :
> Topk first for quick checking
> Run the shell file for each model (This will run run three seeds of the specific model and then take the simple average of the seeds)

2. PaddlePaddle / E:
> Ernie is written in PaddlePaddle and makes for a bit more complicated running; It is however the best performing model of all (by about 2% absolute RCAC on the HM challenge)
> Run ERV-S shell 
> Run ERV-L shell

### Post-processing & ensembling

1. Ensembling
> Place all files in data (Should be 3 csv's per model (dev, test, test_unseen) for a total of 21 csvs) - > Run ens.sh 