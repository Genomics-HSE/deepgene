import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from models import common


class ReformerLabeler(common.MyModule):
    def __init__(self, config, predictor):
        super().__init__()
        #self.save_hyperparameters()
        
        self.reformer = transformers.ReformerModel(config)
        self.predictor = predictor
    
    def forward(self, X):
        output = self.reformer(X)
        assert len(output) == 1
        output = self.predictor(output[0])
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
