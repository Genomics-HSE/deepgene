import functools
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base_models import CategoricalModel, OrdinalModel, ConvEmbedding, NoEmbedding, Predictor
from deepgen.loss import KLDivLoss, CrossEntropyLoss, EMD_squared_loss, CTC_loss, MYLOSS

patch_typeguard()


class LstmLabeler(CategoricalModel):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding],
                 n_class: int,
                 lstm: nn.LSTM,
                 predictor: Predictor,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.n_class = n_class
        self.embedding = embedding
        self.predictor = predictor
        self.lstm = lstm
        self.loss1 = functools.partial(KLDivLoss, n_class)

    @typechecked
    def forward(self, X: TensorType["batch", "genome_length"]) -> TensorType["batch", "genome_length", "hidden_size"]:
        X = self.embedding(X)
        output, _ = self.lstm(X)
        output = self.predictor(output)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    @property
    def name(self) -> str:
        return "LSTM" + "-" + self.embedding.name


