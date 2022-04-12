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


class GruLabeler(CategoricalModel):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding],
                 n_class: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 predictor: Predictor,
                 device: str):
        super().__init__()
        self.n_class = n_class
        self.embedding = embedding
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.1)
        self.predictor = predictor
        
        # self.loss = MYLOSS(n_class, device)
        # self.loss = functools.partial(EMD_squared_loss, n_class)
        # self.loss = CrossEntropyLoss
        self.loss = functools.partial(KLDivLoss, n_class)
    
    @typechecked
    def forward(self, X: TensorType["batch", "genome_length"]) -> TensorType["batch", "genome_length", "hidden_size"]:
        X = self.embedding(X)
        output, _ = self.gru(X)
        output = self.predictor(output)
        return output
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
    
    @property
    def name(self) -> str:
        return "GRU"


class GruLabelerOrdinal(OrdinalModel):
    def __init__(self, labeler, ordinal_head):
        super().__init__()
        self.labeler = labeler
        self.ordinal_head = ordinal_head
        self.n_class = ordinal_head.n_class
    
    def forward(self, X):
        output = self.labeler(X)
        output = self.ordinal_head(output)
        return output
    
    @property
    def name(self):
        return "GRU-Ordinal"
