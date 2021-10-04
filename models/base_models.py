import torch
from pytorch_lightning import LightningModule
from os.path import join
import torch.nn.functional as F
from matplotlib import pyplot as plt
from .viz import make_coalescent_heatmap


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
        #logits = logits.permute(0, 2, 1)
        loss = self.loss(logits, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        
        Y_pred = F.softmax(logits, dim=-1)
        Y_pred = Y_pred.squeeze(0)
        Y = y_batch.squeeze(0)
        
        step = 10000
        # print(X_batch.squeeze(0))
        for j in range(0, len(Y), step):
            f = make_coalescent_heatmap("", (Y_pred[j:j + step].T, Y[j:j + step]))
            plt.show()
        return


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
