import os
from typing import Union, List, Optional
from itertools import cycle

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader


class DatasetPL(pl.LightningDataModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 seq_len: int,
                 n_class: int,
                 batch_size: int,
                 num_workers: int
                 ):
        super(DatasetPL, self).__init__()
        
        self.seq_len = seq_len
        self.n_class = n_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          collate_fn=self.genomes_collate,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          )
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          collate_fn=self.genomes_collate,
                          batch_size=1,
                          num_workers=self.num_workers,
                          )
    
    def genomes_collate(self, batch):
        X = [item[0] for item in batch]
        Y = [item[1] for item in batch]
        X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)
        return X, Y


class DatasetTorch(data.IterableDataset):
    def __init__(self, data_generator):
        super(DatasetTorch, self).__init__()
        self.data_generator = data_generator
    
    def __iter__(self):
        return self.data_generator


def one_hot_encoding_numpy(y_data, num_class):
    return (np.arange(num_class) == y_data[..., None]).astype(np.float32)


if __name__ == '__main__':
    from data_gen_np import get_liner_generator
    
    generator = get_liner_generator(5, 7, 1)
    
    dataset = DatasetTorch(generator)
    
    # for x, y in dataset:
    #     print(x, y)
    
    loader = DataLoader(dataset, batch_size=2)
    for x, y in loader:
        print(x)
