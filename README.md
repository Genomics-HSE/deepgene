# Deepgen library

**Authors**: Kenenber Arzymatov, Evgeniy Khomutov and Vladimir Schur

<!---->

Deepgen is a collection of deep learning models for genomics researchers. It is aimed to speed up a working process for 
people who are not fluent in Deep learning and who want to play around applying different models for their tasks.  

#### Table of Contents

[TOC]

## Run it 

```bash
python main.py --gin_file_train=configs/gru/gru_train.gin \
--gin_file_model=configs/gru/gru_model.gin \
--gin_file_data=configs/gru/gru_data.gin \
--action=train
```