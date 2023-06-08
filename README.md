<p align="center">
    <br>
    <h1 align="center"> 🥶VILIO🥶 </h1> 
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Transformers Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p> State-of-the-art Visio-Linguistic Models 🥶
</h3>

## Updates

### 06/2021 - Hateful Memes CSV Files

- The CSV files that were used for the scores in the <a href="https://arxiv.org/abs/2012.07788">vilio paper</a> are now available <a href="https://www.kaggle.com/muennighoff/vilioresults">here</a>

### 06/2021 - Inference on any meme

- Thanks to the initiative by <a href="https://github.com/katrinc">katrinc</a>, here are two notebooks for using Vilio to perform pure inference on any meme you want :)
- Just adapt the example input dataset / input model to use a different meme / pretrained model🥶
- GPU: https://www.kaggle.com/muennighoff/vilioexample-nb
- CPU: https://www.kaggle.com/muennighoff/vilioexample-nb-cpu


## Ordering

Vilio aims to replicate the organization of huggingface's transformer repo at:
https://github.com/huggingface/transformers

- /bash
Shell files to reproduce hateful memes results

- /data
By default, directory for loading in data & saving checkpoints

- /ernie-vil
Ernie-vil sub-repository written in PaddlePaddle

- /fts_lmdb
Scripts for handling .lmdb extracted features

- /fts_tsv
Scripts for handling .tsv extracted features

- /notebooks
Jupyter Notebooks for demonstration & reproducibility

- /py-bottm-up-attention
Sub-repository for tsv feature extraction forked & adapted from [here](https://github.com/airsplay/py-bottom-up-attention)

- src/vilio
All implemented models (also see below for a quick overview of models)

- /utils
Pandas & ensembling scripts for data handling

- entry.py files
Scripts used to access the models and apply model-specific data preparation

- pretrain.py files
Same purpose as entry files, but for pre-training; Point of entry for pre-training

- hm.py
Training code for the hateful memes challenge; Main point of entry

- param.py
Args for running hm.py


## Usage

Follow SCORE_REPRO.md for reproducing performance on the Hateful Memes Task. <br>
Follow GETTING_STARTED.md for using the framework for your own task. <br>
See the paper at: https://arxiv.org/abs/2012.07788

## Architectures

🥶 Vilio currently provides the following architectures with the outlined language transformers:

1. **[E - ERNIE-VIL](https://arxiv.org/abs/2006.16934)** [ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph](https://arxiv.org/abs/2006.16934)
    - [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)
1. **[D - DeVLBERT](https://arxiv.org/abs/2008.06884)** [DeVLBert: Learning Deconfounded Visio-Linguistic Representations](https://arxiv.org/abs/2008.06884)
    - [BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
1. **[O - OSCAR](https://arxiv.org/abs/2004.06165)** [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165)
    - [BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
1. **[U - UNITER](https://arxiv.org/abs/1909.11740)** [UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/abs/1909.11740)
    - [BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
    - [RoBERTa: Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
1. **[V - VisualBERT](https://arxiv.org/abs/1908.03557)** [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557)
    - [ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942)
    - [BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
    - [RoBERTa: Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
1. **[X - LXMERT](https://arxiv.org/abs/1908.07490)** [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
    - [ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942)
    - [BERT: Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
    - [RoBERTa: Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)


## To-do's

- [ ] Clean-up import statements, python paths & find a better way to integrate transformers (Right now all import statements only work if in main folder)
- [ ] Enable loading and running models just via import statements (and not having to clone the repo)
- [ ] Find a way to better include ERNIE-VIL in this repo (PaddlePaddle to Torch?)
- [ ] Move tokenization in entry files to model-specific tokenization similar to transformers


## Attributions

The code heavily borrows from the following repositories, thanks for their great work:
- https://github.com/huggingface/transformers
- https://github.com/facebookresearch/mmf
- https://github.com/airsplay/lxmert

## Citation

```bibtex
@article{muennighoff2020vilio,
  title={Vilio: State-of-the-art visio-linguistic models applied to hateful memes},
  author={Muennighoff, Niklas},
  journal={arXiv preprint arXiv:2012.07788},
  year={2020}
}
```
