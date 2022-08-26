# Deepgen library

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
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

**Authors**: Kenenber Arzymatov, Evgeniy Khomutov and Vladimir Schur

<!---->

Deepgene is a collection of deep learning models and common datasets for genomics researchers. It is aimed to speed up a working process for 
people who are not fluent in Deep Learning and who want to play around applying different models for their tasks.  

#### Table of Contents

[TOC]
## Motivation 

## Installation

## Quick tour

## Model architectures

## Datasets

## Config files 

Three configuration files has to supplied for training DL models: 
<ul>
  <li>model.gin</li>
  <li>data.gin</li>
  <li>train.gin</li>
</ul>

In `model.gin` all necessary parameters heading by a line `import deepgene.models` for a creation of a choosen model 
should be provided. 



Example of such `.gin` files can be found in `configs` directory.

## Learn more

## Run it 

```bash
python main.py --gin_file_train=configs/gru/gru_train.gin \
--gin_file_model=configs/gru/gru_model.gin \
--gin_file_data=configs/gru/gru_data.gin \
--action=train
```