# Deepgene library

<p align="center">
    <a href="https://circleci.com/gh/Genomics-HSE/deepgene">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/Genomics-HSE/deepgene/main">
    </a>
    <a href="https://github.com/Genomics-HSE/deepgene/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Genomics-HSE/deepgene.svg?color=blue">
    </a>
    <a href="https://github.com/Genomics-HSE/deepgene/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Genomics-HSE/deepgene.svg">
    </a>
    <a>
        <img alt="GitHub downloads" src="https://img.shields.io/github/downloads/genomics-hse/deepgene/total">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

**Authors**: Kenenber Arzymatov, Evgeniy Khomutov and Vladimir Schur

<!---->
#### Table of Contents

[TOC]
## Motivation 

Deepgene is a collection of deep learning models and common datasets for genomics researchers. It is aimed to speed up a working process for 
people who are not fluent in Deep Learning and who want to play around applying different models for their tasks. You can
find a list of available models in [this chapter](https://github.com/Genomics-HSE/deepgene#model-architectures) and
[this chapter](https://github.com/Genomics-HSE/deepgene#datasets). 

## Installation
### Requirements
This project on a deep learning part relies on `pytorch_lightning` library and for a generating genomics data relies on 
`msprime`. 

## Quick tour

## Model architectures

## Datasets

## Config files 

Usually for training DL models big amount of parameters have to be set. You can do it by hard-coding them directly inside
your code, but it is not a good practice. The better way is to use separate files which contain 
configuration parameters outside your main code.  In our framework three configuration files could to be provided: 
<ul>
  <li>`model.gin`</li>
  <li>`data.gin`</li>
  <li>`train.gin`</li>
</ul>

All necessary parameters for a creation of a chosen model, a dataset and a trainer should be provided respectively in each file. 

```
python main.py --model=configs/gru/model.gin --data==configs/gru/data.gin train=configs/gru/train.gin train
```

or 

```
python main.py --model=configs/gru/model.gin --data==configs/gru/data.gin train=configs/gru/train.gin test
```

Example of such `.gin` files can be found in `configs` directory.

## Visualization  

### Comet-ml 


## Learn more
<a href="https://circleci.com/gh/Genomics-HSE/deepgen">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/Genomics-HSE/deepgen/main">
    </a>
