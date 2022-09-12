import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from .base_models import CategoricalModel, OrdinalModel

from deepgene.loss import CrossEntropyLoss, KLDivLoss, EMD_squared_loss, FocalLoss


class ReformerLabeler(transformers.ReformerModel, CategoricalModel):
    def __init__(self, embedding, config, predictor):
        super().__init__(config=config)
        print(type(embedding))
        print(type(config))
        print(type(predictor))
        self.embedding = embedding
        self.predictor = predictor
        
        # self.loss = functools.partial(KLDivLoss, 32)
        # self.loss = functools.partial(EMD_squared_loss, 32)
        self.loss = CrossEntropyLoss
    
    def forward(self, X):
        output = self.embedding(X)
        output = super().forward(
            input_ids=None,
            inputs_embeds=output
        )
        output = self.predictor(output[0])
        return output
    
    @property
    def name(self):
        return "RM"


class ReformerPreTrainerLM(transformers.ReformerForMaskedLM):
    def __init__(self, embedding, config):
        super(ReformerPreTrainerLM, self).__init__(config=config)
        self.masking = Masking(value=2, prob=0.15)
        self.embedding = embedding
        
        self.loss = FocalLoss(gamma=2, alpha=0.002)
    
    def forward(self, X):
        output = self.masking(X)
        output = self.embedding(output)
        output = super().forward(input_ids=None,
                                 inputs_embeds=output)
        return output[0]
    
    def training_step(self, batch, batch_ix):
        X_batch, _ = batch
        labels = torch.clone(X_batch)
        logits = self.forward(X_batch)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
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


class ReformerLabelerOrdinal(OrdinalModel):
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
