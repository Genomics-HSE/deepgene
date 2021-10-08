import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from models import base_models

from .losses import KLDivLoss, CrossEntropyLoss, EMD_squared_loss, CTC_loss


class GruLabeler(base_models.CategoricalModel):
    def __init__(self, embedding, n_class, hidden_size, num_layers, predictor):
        super().__init__()
        self.n_class = n_class
        self.embedding = embedding
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          dropout=0.1)
        self.predictor = predictor
        
        self.loss = CTC_loss
        # self.loss = functools.partial(EMD_squared_loss, n_class)
        # self.loss = CrossEntropyLoss
        # self.loss = functools.partial(KLDivLoss, n_class)
    
    def forward(self, X):
        output = self.embedding(X)
        output, _ = self.gru(output)
        output = self.predictor(output)
        return output
    
    @property
    def name(self):
        return "GRU"


class GruLabelerOrdinal(base_models.OrdinalModel):
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
