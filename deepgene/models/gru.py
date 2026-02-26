import functools
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .base_models import CategoricalModel, OrdinalModel, ConvEmbedding, NoEmbedding, Predictor
from deepgene.loss import KLDivLoss, CrossEntropyLoss, EMD_squared_loss, CTC_loss, MYLOSS, L1Loss, NeLoss, CEandEMD, HuberLoss, NllLoss, CDW_CELoss

patch_typeguard()


class GruLabeler(CategoricalModel):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding],
                 n_class: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 predictor: Predictor,
                 loss: str = "CrossEntropyLoss",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['embedding', 'predictor'])
        self.n_class = n_class
        self.embedding = embedding
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.1)
        self.predictor = predictor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # resolve loss from config
        if isinstance(loss, str):
            if loss == "CrossEntropyLoss":
                self.loss1 = CrossEntropyLoss
            elif loss == "KLDivLoss":
                self.loss1 = functools.partial(KLDivLoss, n_class)
            elif loss == "EMD_squared_loss":
                self.loss1 = functools.partial(EMD_squared_loss, n_class)
            elif loss == "L1Loss":
                self.loss1 = functools.partial(L1Loss, n_class)
            elif loss == "HuberLoss":
                self.loss1 = functools.partial(HuberLoss, n_class)
            elif loss == "NllLoss":
                self.loss1 = NllLoss
            elif loss == "CEandEMD":
                self.loss1 = functools.partial(CEandEMD, n_class)
            elif loss == "CDW_CELoss":
                self.loss1 = CDW_CELoss(num_classes=n_class)
            else:
                # default fallback
                self.loss1 = CrossEntropyLoss
        else:
            # allow passing callable directly via gin
            self.loss1 = loss

    @typechecked
    def forward(self, X: TensorType["batch", "genome_length"]) -> TensorType["batch", "genome_length", "hidden_size"]:
        X = self.embedding(X)
        output, _ = self.gru(X)
        output = self.predictor(output)
        return output
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    @property
    def name(self) -> str:
        return "GRU" + "-" + self.embedding.name


class Mock(CategoricalModel):
    def __int__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


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
