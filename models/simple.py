import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from models import base_models

from .losses import KLDivLoss, CrossEntropyLoss, EMD_squared_loss, CTC_loss, MYLOSS


class SimpleLabeler(base_models.CategoricalModel):
    def __init__(self, n_class, slider, predictor):
        super().__init__()
        self.n_class = n_class
        
        self.slider = slider
        self.linear = nn.Linear(len(slider.windows), n_class)
        self.predictor = predictor
        
        #self.loss = CrossEntropyLoss
        self.loss = functools.partial(EMD_squared_loss, n_class)

    def forward(self, X):
            output = self.slider(X)
            output = self.linear(output)
            output = self.predictor(output)
            return output
    
    @property
    def name(self):
        return "Simple"


class WindowSlider(LightningModule):
    def __init__(self, windows):
        super().__init__()
        self.windows = windows
    
    def forward(self, X):
        X = X.float()
        out = []
        for window_size in self.windows:
            pad = int((window_size - 1) / 2)
            X_temp = F.pad(X, pad=(pad, pad), value=0)
            X_temp = X_temp.unfold(dimension=-1, size=window_size, step=1)
            
            out.append(torch.mean(X_temp, dim=-1))
        
        out = torch.stack(out, dim=2)
        
        return out