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


class ReformerPreTrainerLM(base_models.MLMTrainer):
    def __init__(self, embedding, config):
        super(ReformerPreTrainerLM, self).__init__()
        self.masking = Masking(value=2, prob=0.15)
        self.embedding = embedding
        self.reformer_lm = transformers.ReformerForMaskedLM(config)
        
        self.loss = CrossEntropyLoss
    
    def forward(self, X):
        output = self.masking(X)
        output = self.embedding(output)
        output = self.reformer_lm(
            input_ids=None,
            inputs_embeds=output
        )
        return output[0]
    
    @property
    def name(self):
        return "RM-pretrained"


class Masking(LightningModule):
    def __init__(self, value, prob=0.15):
        super(Masking, self).__init__()
        self.prob = prob
        self.value = value
    
    def forward(self, X):
        rand = torch.rand(X.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < self.prob

        selection = []
        for i in range(X.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(X.shape[0]):
            X[i, selection[i]] = self.value
    
        return X


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
