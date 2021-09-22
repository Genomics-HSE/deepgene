import torch
from pytorch_lightning import LightningModule
from os.path import join
import torch.nn.functional as F


class MyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.loss = torch.nn.BCELoss()  # torch.nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        
        y_batch_ordinal = self.ordinal_transform(y_batch, 32)
        loss = self.loss(logits, y_batch_ordinal)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss,
                }
    
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        preds = torch.exp(logits)
        preds = torch.flatten(preds, start_dim=0, end_dim=1)
        
        # y_batch = torch.argmax(y_batch, dim=-1)
        y = y_batch.flatten()
        
        preds = preds.cpu().detach()
        self.logger.log_coalescent_heatmap(self.name, [preds.T, y.T], batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def save(self, trainer, checkpoint_path):
        print('saving to {parameters_path}'.format(parameters_path=checkpoint_path))
        trainer.save_checkpoint(filepath=checkpoint_path)

    def ordinal_transform(self, y_data, num_classes):
        # y_data (batch_size, seq_len)
        return (y_data[:, :, None] > torch.arange(num_classes-1)).type(torch.FloatTensor)
