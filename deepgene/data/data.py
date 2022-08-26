import os
import sys
from typing import Union, List, Optional
from itertools import cycle
from collections.abc import Callable

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader


class DatasetXY(pl.LightningDataModule):
    def __init__(self,
                 train_generator,
                 val_generator,
                 test_generator,
                 batch_size: int,
                 num_workers: int,
                 ):
        super(DatasetXY, self).__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = DatasetTorch(train_generator)
        self.val_dataset = DatasetTorch(val_generator)
        self.test_dataset = DatasetTorch(test_generator)
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          collate_fn=collate_xy,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset,
                          collate_fn=collate_xy,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          )
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          collate_fn=collate_xy,
                          batch_size=1,
                          num_workers=self.num_workers,
                          )
    

def collate_xy(batch):
    X = [torch.Tensor(item[0]) for item in batch]
    Y = [torch.Tensor(item[1]) for item in batch]
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)
    return X, Y


def collate_xyz(batch):
    X = [torch.Tensor(item[0]) for item in batch]
    Y = [torch.Tensor(item[1]) for item in batch]
    Z = [torch.Tensor(item[2]) for item in batch]
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)
    Z = torch.stack(Z, dim=0)
    return X, Y, Z



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
    
    generator = get_liner_generator(num_genomes=2,
                                    genome_length=10,
                                    num_generators=2)
    
    dataset = DatasetTorch(generator)
    
    # for x, y in dataset:
    #     print(x, y)
    
    loader = DataLoader(dataset, batch_size=2)
    for x, y, z in loader:
        print(len(x))
        print(len(y))
        print(len(z))
