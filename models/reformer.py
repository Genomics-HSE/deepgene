import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from models import base_models

from .losses import CrossEntropyLoss, KLDivLoss, EMD_squared_loss


class ReformerLabeler(base_models.CategoricalModel):
    def __init__(self, embedding, config, predictor):
        super().__init__()
        self.embedding = embedding
        self.reformer = transformers.ReformerModel(config)
        self.predictor = predictor
    
        # self.loss = functools.partial(KLDivLoss, 32)
        # self.loss = functools.partial(EMD_squared_loss, 32)
        self.loss = CrossEntropyLoss

    def forward(self, X):
            output = self.embedding(X)
            output = self.reformer(
                input_ids=None,
                inputs_embeds=output
            )
            output = self.predictor(output[0])
            return output
    
    @property
    def name(self):
        return "RM"


class ReformerLabelerOrdinal(base_models.OrdinalModel):
    def __init__(self, reformer_model, ordinal_head):
        super().__init__()
        self.reformer_model = reformer_model
        self.ordinal_head = ordinal_head
    
    def forward(self, X):
        output = self.reformer_model(X)
        assert len(output) == 1
        output = self.ordinal_head(output)
        return output
    
    @property
    def name(self):
        return "RM-ordinal"
