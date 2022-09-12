# Deepgene library

<p align="center">
    <a href="https://circleci.com/gh/Genomics-HSE/deepgene">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/Genomics-HSE/deepgene">
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

Deepgene is a collection of deep learning models and population demography datasets for genomics researchers. It is aimed to speed up a working process for 
people who are not fluent in Deep Learning and who want to play around applying different models for their tasks. You can
find a list of available models in [this chapter](https://github.com/Genomics-HSE/deepgene#model-architectures) and
[this chapter](https://github.com/Genomics-HSE/deepgene#datasets). 

## Installation
### Requirements
This project on a deep learning part relies on `pytorch_lightning` library and for a generating genomics data relies on 
`msprime`. 

## Quick tour
### Dummy dataset 

For example, you have a classification task when you need to learn a mapping from `X -> y`.
Your `X` data could look like `np.random.rand(1, 3)`, i.e. an array of three numbers, and
`y` just a random true value `np.random.randint(0, 2)`. To create a dataset that can be used
for training a neural network several steps have to be accomplished:

- Create a `generator` python object that can supply a data instances:
```python
import numpy as np
def dummy_generator(shape=(1, 3)):
    while True:
        yield np.random.rand(*shape), np.random.randint(0, 2)

generator = dummy_generator()
X, y = next(generator)
```

- In ML tasks you need three types of datasets: train, validation and test one:
```python
train_generator = dummy_generator()
val_generator = dummy_generator()
test_generator = dummy_generator()
```

- To finally set up a data object you need to create a DatasetXY object: 
```python
data_module = DatasetXY(
        train_generator=train_generator,
        val_generator=val_generator,
        test_generator=test_generator,
        batch_size=8,
        num_workers=8
    )
```

`batch_size` parameter controls how many data instances are fed to a model at once, `num_workers` controls 


### Dummy model 

The simplest model you can use for demonstration purposes is a model that produces the same output as a given input. 
```python 
from deepgene.models import DummyModel
model = DummyModel()
```

### Trainer

The full list of training parameters are available on pytorch-lightning official [documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api). 


### Fit a model 


### Use config files

## Model architectures

## Datasets




## Config files 

Usually for training DL models big amount of parameters have to be set. You can do it by hard-coding them directly inside
your code, but it is not a good practice. The better way is to use separate files which contain 
configuration parameters outside your main code.  In our framework three configuration files could to be provided: 
<ul>
  <li> model.gin </li>
  <li> data.gin </li>
  <li> train.gin </li>
</ul>

All necessary parameters for a creation of a chosen model, a dataset and a trainer should be provided respectively in each file. 

```
python main.py --model=configs/min-config-gru/model.gin --data=configs/min-config-gru/data.gin --train=configs/min-config-gru/train.gin fit```

or 

```
python main.py --model=configs/min-config-gru/model.gin --data=configs/min-config-gru/data.gin --train=configs/min-config-gru/train.gin test
```

Example of such `.gin` files can be found in `configs` directory.

## Visualization  

### Comet-ml 


## Learn more
