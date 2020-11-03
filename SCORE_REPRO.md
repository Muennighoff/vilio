
# Outline to reproduce the ROC-AUC score on Hateful Memes

For the purpose of having everything in one repo, I combined three separate repo's here, which should be treated as separate (in terms of package requirements, as they conflict):
- py-bottom-up (Only for the purpose of feature extraction)
- ernie-vil (For running E-Models)
- vilio (For X, D, V, U, O-Models & everything else)

The pipeline to reproduce the roc-auc score on the public & private leaderboard from scratch on the Hateful Memes challenge follows. If you want to use the pre-trained models and perform inference only, scroll to the end.

# Soft- & Hardware

I used one NVIDIA Tesla P100, Cuda 10.2 & Python 3 for all purposes.
A better GPU/multiple-GPUs will significantly speed up things, but I made sure everything works with just those basics. 
We will install specific packages for each subprocess as outlined below. 


## Data preparation

1. **Images**
- We perform feature extraction before starting to train to speed up training (I also have the code for feature extraction on the go if ever needed (will double training time though)).

Refer to the feature_extraction notebook under notebooks if you run into any problems:

a) Clone the repo;
`git clone https://github.com/Muennighoff/vilio.git`

b) Setup extraction:
`cd vilio/py-bottom-up-attention; pip install -r requirements.txt`
`pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
`cd vilio/py-bottom-up-attention; python setup.py build develop`

This will take a couple of minutes, if you run into any problems, refer to the README under vilio/py-bottom-up-attention/README.md or the notebook. 

c) Place the img folder of the HM-challenge in `vilio/py-bottom-up-attention/data`; Make sure to place the whole img folder in there, i.e. the images will be at vilio/`py-bottom-up-attention/data/img/`

d) Extract features - We use 6 different features for diversity in our model. Run them all at once with:
`cd vilio/py-bottom-up-attention; bash hm_feats.bash` - > TEST!

Alternatively run them individually with the commands in hm_feats.bash, e.g.:

`cd vilio/py-bottom-up-attention; python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr --minboxes 36 --maxboxes 36`

On the hardware I used, each extraction takes about 90 minutes. 

After completing extraction, we have the following files in vilio/py-bottom-up-attention/data:
- hm_vgattr3636.tsv
- hm_vgattr5050.tsv
- hm_vgattr7272.tsv
- hm_vgattr10100.tsv
- hm_vg5050.tsv
- hm_vg10100.tsv

We also make use of the lmdb feats provided by mmf for one of our models, download them at:
https://dl.fbaipublicfiles.com/mmf/data/datasets/hateful_memes/defaults/features/features.tar.gz

With the downloaded detectron.lmdb, we now have 7 different feature files.

For our PyTorch models in the second part, place the following in `vilio/data`:
- hm_vgattr3636.tsv
- hm_vgattr5050.tsv
- hm_vgattr7272.tsv
- hm_vg5050.tsv
- /img/ folder

For our ERNIE-model make copies of the files where necessary and place the following in `vilio/ernie-vil/data/hm`:
- hm_vgattr3636.tsv
- hm_vgattr7272.tsv
- hm_vgattr10100.tsv
- hm_vg5050.tsv
- hm_vg10100.tsv
- /img/ folder


2. **Text** 
- Download the updated train.jsonl, dev_seen.jsonl, dev_unseen.jsonl, test_seen.jsonl, test_unseen.jsonl from the HM Challenge and place them in BOTH the data folder under `vilio/data` and the data folder under `vilio/ernie-vil/data/hm`


## Individual Model Pretraining & Training

1. PyTorch / D O U V X :
Make sure we have 5 jsonl files, 4 tsv files, 1 lmdb file and 1 img folder under vilio/data.
Install the necessary packages with: 
`cd vilio; pip install -r requirements_full.txt` - > TEST; Generate with pipx

We now proceed to the training of our models. For all models we make use of pre-trained models provided by the creators of each model. I will put the original DL-Links for all models below, but I have reuploaded all model weights to datasets on kaggle, so contact me in case any of the original DL-Links does not work anymore.

O, V & X are first pretrained with combinations of MaskedLM and ITM (Image-Text-Matching) on the previously extracted features. We then construct a clean train-file by dropping some duplicates and adding the data from dev_unseen.jsonl that is not in dev_seen.jsonl. D, O, U, V, X are then trained on the clean train-file and validated using dev_seen.jsonl. Using a checkpoint saved in the middle of training we re-train each model on subparts of the data. The same is repeated without validation on the clean train-file and dev_seen to produce test and test_unseen estimates. We use the dev_seen estimates to optimize & combine all predictions and end up with 3 csv files for each model: dev, test, test_unseen

- D-Model:
Download the pre-trained model [here](https://drive.google.com/file/d/151vQVATAlFM6rs5qjONMnIJBGfL8ea-B/view?usp=sharing) and place the file pytorch_model_11.bin under `vilio/data`
Run `cd vilio; bash /bash/hm_D.bash`.

- O-Model:
Download the pre-trained model [here](https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/large-vg-labels.zip), unzip it and take /large-vg-labels/ep_20_590000/pytorch_model.bin, and place the file pytorch_model.bin under `vilio/data`.
Run `cd vilio; bash /bash/hm_O.bash`.

- U-Model:
Download the pre-trained model [here](https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-large.pt) and place the file uniter-large.pt under `vilio/data`.
Run `cd vilio; bash /bash/hm_U.bash`.

- V-Model:
Download the pre-trained model [here](https://dl.fbaipublicfiles.com/mmf/data/models/visual_bert/visual_bert.pretrained.coco.tar.gz) and place the file model.pth under `vilio/data`.

- X-Model:
Download the pre-trained model [here](http://nlp.cs.unc.edu/models/lxr1252_bertinit/Epoch18_LXRT.pth) and place the file Epoch18_LXRT.pth under `vilio/data`.



The following does all of the above for each model. We will end up with three csv's per model in data:
`cd vilio; bash /bash/hm_D.bash`
`cd vilio; bash /bash/hm_O.bash`
`cd vilio; bash /bash/hm_U.bash`
`cd vilio; bash /bash/hm_V.bash`
`cd vilio; bash /bash/hm_X.bash`

https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-large.pt

> Topk first for quick checking
> Run the shell file for each model (This will run run three seeds of the specific model and then take the simple average of the seeds)

2. PaddlePaddle / E:
Make sure we have 5 jsonl files, 5 tsv files and 1 img folder under vilio/data.

- E - Large:
Download the pre-trained models here and here.

- E - Small:
Download the pre-trained models here and here.

> Ernie is written in PaddlePaddle and makes for a bit more complicated running; It is however the best performing model of all (by about 2% absolute RCAC on the HM challenge)
> Run ERV-S shell 
> Run ERV-L shell

## Post-processing & ensembling

1. Ensembling
> Place all files in data (Should be 3 csv's per model (dev, test, test_unseen) for a total of 21 csvs) - > Run ens.sh 



## Inference-only

Following features:
- Reextract?


Following weights:
- Re-train all using vilio? 


# > Create a notebook with everything in one