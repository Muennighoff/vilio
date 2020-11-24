
# Outline to reproduce the ROC-AUC score on Hateful Memes

Below follows an overview of the repo and the full outline to reproduce the results on the Hateful Memes Challenge hosted by Facebook & DrivenData. If you run into any issue (no matter how small) do send me an email at n.muennighoff@gmail.com. Also contact me / open a PR if you have improvements for the structure or code of this repository. 

For the purpose of having everything in one repo, I combined three separate repo's here, which should be treated as separate (in terms of package requirements, as they conflict):
- py-bottom-up-attention (Only for the purpose of feature extraction)
- ernie-vil (For E-Models)
- vilio (For O, U, V-Models & everything else)

The pipeline to reproduce the roc-auc score on the public & private leaderboard from scratch on the Hateful Memes challenge follows. If you want to use the pre-trained models and perform inference only, scroll to the end.

## Soft- & Hardware

I used one NVIDIA Tesla P100, Cuda 10.2, a Linux environment & Python 3 for all purposes.
A better GPU/multiple-GPUs will significantly speed up things, but I made sure everything works with just those basics. 
Each of the subrepos has its own requirements.txt which can be installed via:
- `cd vilio/py-bottom-up-attention; pip install -r requirements.txt`
- `cd vilio; pip install -r requirements.txt`
- `cd vilio/ernie-vil; pip install -r requirements.txt`


## Data preparation

### Images
- We perform feature extraction before starting to train to speed up training (I also have the code for feature extraction on the go if ever needed (will double training time though)).

Refer to the feature_extraction notebook under `vilio/notebooks` if you run into any problems:

- Clone the repo: <br>
`git clone https://github.com/Muennighoff/vilio.git`

- Setup extraction: <br>
`cd vilio/py-bottom-up-attention; pip install -r requirements.txt` <br>
`pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'` <br>
`cd vilio/py-bottom-up-attention; python setup.py build develop` <br>
This will take a couple of minutes, if you run into any problems, refer to the README under `vilio/py-bottom-up-attention/README.md` or the notebook. 

- Place the img folder of the HM-challenge in `vilio/py-bottom-up-attention/data` <br>
Make sure to place the whole img folder in there, i.e. the images will be at `vilio/py-bottom-up-attention/data/img/`

- Extract features - We use 6 different features for diversity in our models. Run them all at once with (~6h): <br>
`cd vilio/py-bottom-up-attention; bash hm_feats.bash`

- Alternatively run them individually with the commands in hm_feats.bash, e.g.: <br>
`cd vilio/py-bottom-up-attention; python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split img --weight vgattr --minboxes 36 --maxboxes 36` <br>
On the hardware I used, each extraction takes about 90 minutes. 
<br>

After completing extraction, we have the following files in `vilio/py-bottom-up-attention/data`:
- hm_vgattr3636.tsv
- hm_vgattr5050.tsv
- hm_vgattr7272.tsv
- hm_vgattr10100.tsv
- hm_vg5050.tsv
- hm_vg10100.tsv

We will still need the /img/ folder from HM for the models, as we initally remove some duplicates from the training data for a cleaner dataset. Place the /img/ folder under both `vilio/data` & `vilio/ernie-vil/data/hm`.
We also make use of the lmdb feats provided by mmf for one of our models, download them [here](https://dl.fbaipublicfiles.com/mmf/data/datasets/hateful_memes/defaults/features/features.tar.gz).
<br>
With the downloaded detectron.lmdb, we now have 7 different feature files.

For the PyTorch models in the second part, place the following in `vilio/data`:
- hm_vgattr3636.tsv
- hm_vgattr5050.tsv
- hm_vgattr7272.tsv
- hm_vg5050.tsv
- detectron.lmdb
- /img/ folder


For the ERNIE-models make copies of the files where necessary and place the following in `vilio/ernie-vil/data/hm`:
- hm_vgattr3636.tsv
- hm_vgattr7272.tsv
- hm_vgattr10100.tsv
- hm_vg5050.tsv
- hm_vg10100.tsv
- /img/ folder


### Text 
- Download the updated train.jsonl, dev_seen.jsonl, dev_unseen.jsonl, test_seen.jsonl, test_unseen.jsonl from the HM Challenge and place them in BOTH the data folder under `vilio/data` and the hm folder under `vilio/ernie-vil/data/hm`


## Individual Model Pretraining, Training & Inference

The below combines both training & inference. For inference-only scroll to the bottom. 
Refer to the hm_pipeline notebook under `vilio/notebooks` for an example of running training & inference for all models. If you have unlimited resources you can just run `vilio/notebooks/hm_pipeline.ipynb` as is and you will have the final csvs to submit to the leaderboard. I also uploaded the notebook to kaggle [here](https://www.kaggle.com/muennighoff/hm-pipeline), where I already downloaded all the pretrained models. In order to run it with limited hard- & software though the models need to be split up.
<br><br>
### 1. PyTorch / O U V
Make sure we have 5 jsonl files, 4 tsv files, 1 lmdb file and 1 img folder under `vilio/data`.
Install the necessary packages with: <br>
`cd vilio; pip install -r requirements_full.txt`

We now proceed to the training of our models. For all models we make use of pre-trained models provided by the creators of each model. I will put the original DL-Links for all models below, but I have reuploaded all model weights to datasets on kaggle, so contact me in case any of the original DL-Links does not work anymore (You can also find the datasets in the hm_pipeline notebook).

O & V are first task-specific pretrained with combinations of MaskedLM and ITM (Image-Text-Matching) on the previously extracted features. We then construct a clean train-file by dropping some duplicates and adding the data from dev_unseen.jsonl that is not in dev_seen.jsonl (When it prints "Preparing..."). O, U & V are then trained on the clean train-file and validated using dev_seen.jsonl. The same is repeated without validation on the clean train-file + dev_seen to produce test_seen and test_unseen estimates on as much data as possible. We use the dev_seen estimates to optimize & combine all 3 seeds per model and end up with 3 csv files for each model: dev_seen, test_seen, test_unseen


- **O-Model:**
Download the pre-trained model [here](https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/large-vg-labels.zip), unzip it and take `/large-vg-labels/ep_20_590000/pytorch_model.bin`, and place the file pytorch_model.bin under `vilio/data/pytorch_model.bin`. The file `vilio/bash/training/O/hm_O.sh` will pretrain the model, run it on three different features & seeds and then simple average them. First to make sure everything works correctly, run the file once with only a tiny part of data by changing the topk argument, for example as follows: `cd vilio; bash bash/training/O/hm_O.sh 20`.This will run the file with only 20 images. If there are any file errors make sure the the necessary files are under `vilio/data` as outlined above. Else raise an issue/send me an email and I will look into it asap. If there are no apparent errors, run `cd vilio; bash bash/training/O/hm_O.sh`. On a P100, each model takes around 3 hours, hence it will run for **~9h**. If that's too long, we also have .sh files separating the steps.
For separating the steps, run `cd vilio; bash bash/training/O/hm_OV50.sh`, `cd vilio; bash bash/training/O/hm_O50.sh` and `cd vilio; bash bash/training/O/hm_O36.sh` (I didn't have the memory to run 72 feats for O, hence we run O50, once with vgattr extraction, once with vg extraction). Then run `cd vilio; bash bash/training/O/hm_OSA.sh` to simple average the 9 csvs that have been created. This will create a new dir within `vilio/data/` named after the experiment (O365050), where it will store the final **3 csvs (dev_seen, test_seen, test_unseen)** for the O-Model, which will go into the final ensemble. If you plan to reset your environment, make sure to save those 3 csvs, everything else can be discarded. 

- **U-Model:**
Download the pre-trained model [here](https://convaisharables.blob.core.windows.net/uniter/pretrained/uniter-large.pt) and place the file uniter-large.pt under `vilio/data`.
We follow the same procedure as for O-Model, except that there will be no pre-training. Run `cd vilio; bash bash/training/U/hm_U.sh` to do all at once. Alternatively `cd vilio; bash bash/training/U/hm_U36.sh`, `cd vilio; bash bash/training/U/hm_U50.sh`, `cd vilio; bash bash/training/U/hm_U72.sh` `cd vilio; bash bash/training/U/hm_USA.sh`. You can run a test with `cd vilio; bash bash/training/U/hm_U.sh 20`. P100 Runtime in total: **~5h**.

- **V-Model:**
Download the pre-trained model [here](https://dl.fbaipublicfiles.com/mmf/data/models/visual_bert/visual_bert.pretrained.coco.tar.gz) untar it and place the file model.pth under `vilio/data`.
For V we will be using PyTorch 1.6 to make use of SWA which is >=1.6. For my setup this means running: `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`. Check out how to install PyTorch 1.6 [here](https://pytorch.org/get-started/previous-versions/). 
We use the lmdb features for V as well as pretraining, so make sure you have installed lmdb as in `vilio/requirements.txt`. After installing PyTorch 1.6, run `cd vilio; bash bash/training/V/hm_V.sh`to extract lmdb feats, run three different seeds and averaging. Alternatively run `cd vilio; bash bash/training/V/hm_VLMDB.sh` to extract lmdb feats and then `cd vilio; bash bash/training/V/hm_V45.sh`, `cd vilio; bash bash/training/V/hm_V90.sh`, `cd vilio; bash bash/training/V/hm_V135.sh`, followed by `cd vilio; bash bash/training/V/hm_VSA.sh`. If you want to test, I recommend just running one of the single bash files with extraction before, which should run in around 1 hour (Modify the epochs parameter to test even quicker). P100 Runtime in total:  **~3h**.

### 2. **PaddlePaddle / E:**
Make sure we have 5 jsonl files, 5 tsv files and 1 img folder under `vilio/ernie-vil/hm/data`. Install the necessary packages with `cd vilio/ernie-vil; pip install -r requirements.txt`. Some of them will install different versions of packages previously installed.

- **E - Large:**
Download the pre-trained model LARGE PRETRAINED [here](https://ernie-github.cdn.bcebos.com/model-ernie-vil-large-en.1.tar.gz). Place the params folder in the folder called "ernielarge" under `vilio/ernie-vil/data/ernielarge` (The params folder only, the other files are already in  `vilio/ernie-vil/data/ernielarge`). Now dowload LARGE VCR FINETUNED [here](https://ernie-github.cdn.bcebos.com/model-ernie-vil-large-VCR-task-pre-en.1.tar.gz) and place the params folder in `vilio/ernie-vil/data/ernielargevcr`. We will be using both the original pre-trained model & the VCR finetuned model, as it increases diversity. Next run `cd vilio/ernie-vil; bash bash/training/EL/hm_EL.sh` On my setup this would run for **~8h**, as it runs 5 different features. If that's too long, you can run `cd vilio/ernie-vil; bash bash/training/EL/hm_EL36.sh`, `cd vilio/ernie-vil; bash bash/training/EL/hm_ELVCR36.sh`, `cd vilio/ernie-vil; bash bash/training/EL/hm_ELV50.sh`, `cd vilio/ernie-vil; bash bash/training/EL/hm_EL72.sh`, `cd vilio/ernie-vil; bash bash/training/EL/hm_ELVCR72.sh`, followed by `cd vilio/ernie-vil; bash bash/training/EL/hm_ELSA.sh`. (Note that we run EL36 & then ELVCR36 as they use the same features hence it saves a copy operation; else make sure to recopy the feats back in).

- **E - Small:**
Download the pre-trained model SMALL PRETRAINED [here](https://ernie-github.cdn.bcebos.com/model-ernie-vil-base-en.1.tar.gz). Place the params folder in the folder `vilio/ernie-vil/data/erniesmall`. Do the same for SMALL VCR PRETRAINED [here](https://ernie-github.cdn.bcebos.com/model-ernie-vil-base-en.1.tar.gz) and place the "params" folder in `vilio/ernie-vil/data/erniesmallvcr/`. Follow the same procedure as for E - Large, running: `cd vilio/ernie-vil; bash bash/training/ES/hm_ES36.sh`, `cd vilio/ernie-vil; bash bash/training/ES/hm_ESVCR36.sh`, `cd vilio/ernie-vil; bash bash/training/ES/hm_ESV50.sh`, `cd vilio/ernie-vil; bash bash/training/ES/hm_ES72.sh`, `cd vilio/ernie-vil; bash bash/training/ES/hm_ESVCR72.sh`, followed by `cd vilio/ernie-vil; bash bash/training/ES/hm_ESSA.sh`. 

I am not yet very advanced in PaddlePaddle, but definitely let me know if there are any issues. When the output ends with _2950 Aborted (core dumped)_ or it says _numpy.multicore array not found_ that is normal. I am using "mv" statements in the shell scripts to move around the features, instead of "cp" to save space, hence in the hm_ES & hm_EL shell files we run the features in order (36, 36, 50, 72, 72), so we can afford to move them and keep them for the next run. (For the 2nd 36 & 72 it will show a .tsv not found error, but thats fine as the tsv is already in place). In general as long as it runs, it is fine. :) 


## Combining

Take the csvs from all models (In their respective experiment folders) and drop them into `vilio/data/`. Make sure you place 3 csvs (dev_seen, test_seen, test_unseen) for each model there. If you ran all 5 models, then you should have 15 csv files in `vilio/data/` (O, U, V, ES, EL). Apart from that, you still want to have the HM data (.jsonls) in `vilio/data/`, as we will use the dev_seen.jsonl for weight optimization.  Run `cd vilio; bash bash/hm_ens.sh`. This will loop through multiple ensembling methods (simple averaging, power averaging, rank averaging, optimization) and output three final csvs in `./vilio/data` starting with FIN_. Submit the test_seen / test_unseen version of those.


## Inference-Only

The above is the full pipeline to train, infer & ensemble. If you want to perform inference only without training, I have uploaded weights & set up an example notebook with the exact inference pipeline that you can find under `vilio/notebooks`. I also uploaded the notebook split into three to kaggle. They already include the weight & feature data as inputs. To run them:
1) [Inference1](https://www.kaggle.com/muennighoff/hm-inference1): Upload/download the hatefulmemes data and copy paste the img folder and the jsonls into `vilio/ernie-vil/data/hm`, as outlined in the notebook. Then commit it (~1h).
2) [Inference2](https://www.kaggle.com/muennighoff/hm-inference2): Upload/download the hatefulmemes data and copy paste the img folder and the jsonls into `vilio/data/ernie-vil/data/hm`, as outlined in the notebook. Then commit it (~1h).
3) [Inference3](https://www.kaggle.com/muennighoff/hm-inference3): Grab the csv files that were output from Inference1 & Inference2 (Click on the committed notebook and scroll to the output part). Upload those 6 csvs as input data to Inference3. Make sure they get copied to `/vilio/data/` in the end, as outlined in the notebook. Also grab the hatefulmemes data and place the img folder & jsonls at `vilio/data` as outlined in the notebook. Commit it (~3h). Take the output test_seen / test_unseen starting with FIN_ and submit them. 
Sometimes cuda throws an initialization error when running many models consecutively. In that case it is best to split them and run them one by one making sure to save the three csvs per model. Inference-only on all seeds & models together takes around 5h, while the full pipeline (incl. training, inference & ensembling) takes around 25h. 

Below is all the pre-trained/extracted data mentioned in the notebooks:

[Extracted TSV Features](https://www.kaggle.com/muennighoff/hmtsvfeats) <br>
[Provided LMDB Features](https://www.kaggle.com/muennighoff/hmfeatureszipfin)

Weights (2 ckpts per run - ignore the sub ckpts):
- [O36](https://www.kaggle.com/muennighoff/vilioo36)
- [O50](https://www.kaggle.com/muennighoff/vilioo50)
- [OV50](https://www.kaggle.com/muennighoff/vilioov50)
- [U36](https://www.kaggle.com/muennighoff/viliou36)
- [U50](https://www.kaggle.com/muennighoff/viliou50)
- [U72](https://www.kaggle.com/muennighoff/viliou72)
- [V45](https://www.kaggle.com/muennighoff/viliov45)
- [V90](https://www.kaggle.com/muennighoff/viliov90)
- [V135](https://www.kaggle.com/muennighoff/viliov135)
- [ES36](https://www.kaggle.com/muennighoff/vilioes36)
- [ESVCR36](https://www.kaggle.com/muennighoff/vilioesvcr36)
- [ESV50](https://www.kaggle.com/muennighoff/vilioesv50)
- [ES72](https://www.kaggle.com/muennighoff/vilioes72)
- [ESVCR72](https://www.kaggle.com/muennighoff/vilioesvcr72)
- [EL36](https://www.kaggle.com/muennighoff/el36viliodev)
- [ELVCR36](https://www.kaggle.com/muennighoff/elvcr36viliodev)
- [ELV50](https://www.kaggle.com/muennighoff/elv50viliodev)
- [EL72](https://www.kaggle.com/muennighoff/el72viliodev)
- [ELVCR72](https://www.kaggle.com/muennighoff/elvcr72viliodev)


## Additional thoughts

Just by e.g. running only one instead of three (or five for E) seeds per model, you can cut down inference time by **80%** without losing more than **absolute 1-3%** on the roc-auc metric. Similarly, using less models significantly speeds up things. In case you want to try that, here is a rough ranking of the best models: **1.EL  2.ES  3.U/O/V** <br> There is also room for optimization in the code (e.g. cp statements; reloading tsv feats every time), with which I am sure we can get inference down to **~10min** and performance dropping less than **absolute 3%** (and I'll be working on this!). I'd love to help on any project trying to decomplexify things & making it production optimized! Also I'd love to hear any critic of the repo. <br> 
Sending lots of love your way!🥶


## Additional Models not used
These models are not used in the original submission, but since I have implemented them here's how to use them :) 

- **D-Model:**
Download the pre-trained model [here](https://drive.google.com/file/d/151vQVATAlFM6rs5qjONMnIJBGfL8ea-B/view?usp=sharing) and place the file pytorch_model_11.bin under `vilio/data/pytorch_model_11.bin`
The file `vilio/bash/training/D/hm_D.sh` will run the model on three different features & seeds and then simple average them. First to make sure everything works correctly, run the file once with only a tiny part of data by changing the topk argument, for example as follows: `cd vilio; bash bash/training/D/hm_D.sh 20`. This will run the file with only 20 images. If there are any file errors make sure the the necessary files are under `vilio/data` as outlined above. Else raise an issue/send me an email and I will look into it asap. If there are no apparent errors, run `cd vilio; bash bash/training/D/hm_D.sh`. On a P100, each model takes around 1 hour, hence it will run for **~4h**. If that's too long, we also have .sh files separating the steps. For separating the steps run `cd vilio; bash bash/training/D/hm_D36.sh`, `cd vilio; bash bash/training/D/hm_D50.sh`, `cd vilio; bash bash/training/D/hm_D72.sh` and then run `cd vilio; bash bash/training/D/hm_DSA.sh` to simple average the the 9 csvs that have been created. This will create a new dir within `vilio/data/` named after the experiment (D365072), where it will store the final **3 csvs (dev_seen, test_seen, test_unseen)** for the D-Model, which will go into the final ensemble. If you plan to reset your environment, make sure to save those 3 csvs, everything else can be discarded.

- **X-Model:**
Download the pre-trained model [here](http://nlp.cs.unc.edu/models/lxr1252_bertinit/Epoch18_LXRT.pth) and place the file Epoch18_LXRT.pth under `vilio/data`.
For X we will be using PyTorch 1.6 to make use of SWA which is >=1.6. For my setup this means running: `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`. Check out how to install PyTorch 1.6 [here](https://pytorch.org/get-started/previous-versions/). We also perform pretraining and use different tsv features. After installing PyTorch 1.6, run `cd vilio; bash bash/training/X/hm_X.sh`to run three different seeds and averaging. Alternatively `cd vilio; bash bash/training/X/hm_X36.sh`, `cd vilio; bash bash/training/X/hm_X50.sh`, `cd vilio; bash bash/training/X/hm_X72.sh` `cd vilio; bash bash/training/X/hm_XSA.sh`.  You can run a test with `cd vilio; bash bash/training/X/hm_X.sh 20`. P100 Runtime in total:  **~5h**.
