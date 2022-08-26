# Deepgen library

**Authors**: Kenenber Arzymatov, Evgeniy Khomutov and Vladimir Schur

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<!---->

Deepgen is a collection of deep learning models and common datasets for genomics researchers. It is aimed to speed up a working process for 
people who are not fluent in Deep Learning and who want to play around applying different models for their tasks.  

#### Table of Contents

[TOC]
## Motivation 

## Installation

## Quick tour  

## Model architectures

## Datasets

## Learn more

## Run it 

```bash
python main.py --gin_file_train=configs/gru/gru_train.gin \
--gin_file_model=configs/gru/gru_model.gin \
--gin_file_data=configs/gru/gru_data.gin \
--action=train
```