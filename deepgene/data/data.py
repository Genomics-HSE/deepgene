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


class DatasetTorch(data.IterableDataset):
    def __init__(self, data_generator):
        super(DatasetTorch, self).__init__()
        self.data_generator = data_generator

    def __iter__(self):
        return self.data_generator


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


def one_hot_encoding_numpy(y_data, num_class):
    return (np.arange(num_class) == y_data[..., None]).astype(np.float32)


def dummy_generator(shape=(1, 3)):
    while True:
        yield np.random.rand(*shape), np.random.randint(0, 2)


class DummyDataset(DatasetXY):
    def __init__(self, data_shape=(1, 3), batch_size=8, num_workers=8):
        train_generator = dummy_generator(data_shape)
        val_generator = dummy_generator(data_shape)
        test_generator = dummy_generator(data_shape)

        super(DummyDataset, self).__init__(
            train_generator=train_generator,
            val_generator=val_generator,
            test_generator=test_generator,
            batch_size=batch_size,
            num_workers=num_workers
        )


if __name__ == '__main__':
    pass
