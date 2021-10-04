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


class ReformerLabeler(base_models.BaseModel):
    def __init__(self, config, predictor, ordinal_head):
        super().__init__()
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




