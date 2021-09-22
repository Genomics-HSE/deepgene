import gin
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from .viz import make_coalescent_heatmap
from matplotlib import pyplot as plt


def train_model(trainer: Trainer,
                model: LightningModule,
                data_module: LightningDataModule,
                checkpoint_path: str,
                resume: bool
                ):
    if resume:
        trainer = Trainer(resume_from_checkpoint=checkpoint_path)
    
    print("Running {}-model...".format(model.name))
    
    trainer.fit(model=model, datamodule=data_module)
    model.save(trainer, checkpoint_path)
    
    return trainer, model


def test_model(model: LightningModule,
               checkpoint_path: str,
               test_output: str,
               datamodule: LightningDataModule,
               ):
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    test_data_loader = datamodule.test_dataloader()
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_data_loader):
            Y_pred = model(X)
            Y = Y.squeeze(0)
            Y_pred = F.softmax(Y_pred.squeeze(0), dim=-1)
            
            step = 20000
            for j in range(0, len(Y), step):
                f = make_coalescent_heatmap("", (Y_pred[j:j+step].T, Y[j:j+step]))
                plt.show()
            return
    return
