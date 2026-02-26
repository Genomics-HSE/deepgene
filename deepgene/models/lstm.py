import functools
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base_models import CategoricalModel, OrdinalModel, ConvEmbedding, NoEmbedding, Predictor
from deepgene.loss import KLDivLoss, CrossEntropyLoss, EMD_squared_loss, CTC_loss, MYLOSS

patch_typeguard()


class LstmLabeler(CategoricalModel):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding],
                 n_class: int,
                 # lstm: nn.LSTM,
                 predictor: Predictor,
                 loss: str = "CrossEntropyLoss",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['embedding', 'predictor'])
        self.n_class = n_class
        self.embedding = embedding
        self.predictor = predictor
        self.lstm = nn.LSTM(
            input_size = 1,
            hidden_size = n_class,
            num_layers = 4,
            batch_first = True,
            dropout = 0.1,
            bidirectional = True
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # resolve loss from config
        if isinstance(loss, str):
            if loss == "CrossEntropyLoss":
                self.loss1 = CrossEntropyLoss
            elif loss == "KLDivLoss":
                self.loss1 = functools.partial(KLDivLoss, self.n_class)
            elif loss == "EMD_squared_loss":
                self.loss1 = functools.partial(EMD_squared_loss, self.n_class)
            else:
                self.loss1 = CrossEntropyLoss
        else:
            self.loss1 = loss

    @typechecked
    def forward(self, X: TensorType["batch", "genome_length"]) -> TensorType["batch", "genome_length", "hidden_size"]:
        X = self.embedding(X)
        output, _ = self.lstm(X)
        output = self.predictor(output)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    @property
    def name(self) -> str:
        return "LSTM" + "-" + self.embedding.name


