<p align="center">
    <br>
    <h1 align="center"> 🥶VILIO🥶 </h1> 
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p> State-of-the-art Vision & Linguistics Order 🥶
</h3>


## Ordering

Vilio aims to replicate the organization of huggingface's transformer repo at:
https://github.com/huggingface/transformers

- src/models
All implemented models (also see below for a quick overview of models)

- /postprocessing
Scripts applied for postprocessing & ensembling in the hateful memes challenge

- /notebooks
Jupyter Notebooks to fully reproduce results

- /tools
Tools that did not add value, but may in the future and hence have been archived for the purpose of structure (e.g. adding textb to OSCAR...)

- entry.py files
Scripts used to access the models and apply model-specific data preparation

- pretrain.py files
Same purpose as entry files, but for pre-training

- hm.py
Training code for the hateful memes challenge

- param.py
Args for running hm.py

- data files
Data loading files specific for the hateful memes challenge (They can be easily adapted to fit other datasets)


## Usage

Follow SCORE_REPROD.md


## Architectures

🥶 Vilio currently provides the following architectures with the outlined language transformers:

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

- [ ] Enable loading and running models just via import statements (and not having to clone the repo)
- [ ] Find a way to include ERNIE-VIL in this repo (PaddlePaddle to Torch?)
- [ ] Move tokenization in entry files to model-specific tokenization similar to transformers


## Attributions

The code heavily borrows from the following repositories:
- https://github.com/huggingface/transformers
- https://github.com/facebookresearch/mmf
- https://github.com/airsplay/lxmert
