import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from models import base_models

from .losses import KLDivLoss


class ReformerLabeler(base_models.BaseModel):
    def __init__(self, config, predictor, ordinal_head):
        super().__init__()
        # self.save_hyperparameters()
        
        self.reformer = transformers.ReformerModel(config)
        self.predictor = predictor
        self.ordinal_head = ordinal_head
    
    def forward(self, X):
        output = self.reformer(X)
        assert len(output) == 1
        output = self.predictor(output[0])
        output = self.ordinal_head(output)
        return output
    
    @property
    def name(self):
        return "RM"


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

        # self.loss = nn.CrossEntropyLoss()
        self.loss = functools.partial(KLDivLoss, n_class)
    
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


class Predictor(LightningModule):
    def __init__(self, d_model, dropout, n_class):
        super().__init__()
        
        self.dense1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_model, n_class)
    
    def forward(self, X):
        output = self.dropout1(F.relu(self.dense1(X)))
        output = self.dense2(output)
        return output


class OrdinalHead(LightningModule):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.n_class = n_class
        self.dense1 = nn.Linear(d_model, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_class - 1))
    
    def forward(self, X):
        output = self.dense1(X)
        output = output + self.bias
        return output


class ConvEmbedding(LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv_emb = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding='same'
        )
    
    def forward(self, X):
        X = X.float()
        X = X.unsqueeze(1)
        output = self.conv_emb(X)
        output = output.permute(0, 2, 1)
        return output
