import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from models import common


class ReformerLabeler(common.MyModule):
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
        self.dense1 = nn.Linear(d_model, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_class - 1))
    
    def forward(self, X):
        output = self.dense1(X)
        output = F.sigmoid(output + self.bias)
        return output
