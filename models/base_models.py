import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from os.path import join
import torch.nn.functional as F
from matplotlib import pyplot as plt
from .viz import make_coalescent_heatmap


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
    
    def save(self, trainer, checkpoint_path):
        print('saving tox {parameters_path}'.format(parameters_path=checkpoint_path))
        trainer.save_checkpoint(filepath=checkpoint_path)


class CategoricalModel(BaseModel):
    def __init__(self):
        super(CategoricalModel, self).__init__()
    
    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        loss = self.loss(logits, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        
        Y_pred = F.softmax(logits, dim=-1)
        Y_pred = Y_pred.squeeze(0)
        Y = y_batch.squeeze(0)
        step = 32768
        # print(X_batch.squeeze(0))
        for j in range(0, len(Y), step):
            f = make_coalescent_heatmap("", (Y_pred[j:j + step].T, Y[j:j + step]))
            plt.show()
        return Y_pred


class MLMTrainer(BaseModel):
    def __init__(self):
        super(MLMTrainer, self).__init__()
    
    def training_step(self, batch, batch_ix):
        X_batch, _ = batch
        labels = torch.clone(X_batch)
        logits = self.forward(X_batch)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss}


class OrdinalModel(BaseModel):
    def __init__(self):
        super(OrdinalModel, self).__init__()
        self.loss = torch.nn.BCELoss()
    
    def ordinal_transform(self, y_data):
        # y_data (batch_size, seq_len)
        return (y_data[:, :, None] > torch.arange(self.n_class - 1)).float()
    
    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        logits = torch.sigmoid(logits)
        y_batch = self.ordinal_transform(y_batch)
        loss = self.loss(logits, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss,
                }
    
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        
        Y_pred = torch.sigmoid(logits)
        Y_pred = Y_pred.squeeze(0)
        Y = y_batch.squeeze(0)
        print(X_batch.squeeze(0))
        step = 10000
        for j in range(0, len(Y), step):
            f = make_coalescent_heatmap("", (Y_pred[j:j + step].T, Y[j:j + step]))
            plt.show()
        return


class Predictor(LightningModule):
    def __init__(self, d_model, dropout, n_class):
        super().__init__()
        
        self.dense1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(d_model, n_class)
    
    def forward(self, X):
        output = self.dropout1(F.relu(self.dense1(X)))
        output = self.dropout2(F.relu(self.dense2(output)))
        output = self.dense3(output)
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
    def __init__(self, n_layers, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        conv_embeds = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding='same'
            )
        ]
        
        conv_embeds.extend(
            [nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='same'
                ) for _ in range(n_layers - 1)]
        )
        
        self.conv_embeds = nn.ModuleList(conv_embeds)
    
    def forward(self, X):
        output = X.float()
        output = output.unsqueeze(1)
        
        for layer in self.conv_embeds:
            output = layer(output)
        
        output = output.permute(0, 2, 1)
        return output


class NoEmbedding(LightningModule):
    def __init__(self):
        super(NoEmbedding, self).__init__()
    
    def forward(self, X):
        return X.unsqueeze(2).float()
