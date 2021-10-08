from matplotlib import pyplot as plt
import gin
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from .reformer import ReformerLabeler
from .gru import GruLabeler
from .viz import make_coalescent_heatmap


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


def test_model(trainer: Trainer,
               model: LightningModule,
               checkpoint_path: str,
               test_output: str,
               datamodule: LightningDataModule,
               ):
    print(checkpoint_path)
    model = ReformerLabeler.load_from_checkpoint(checkpoint_path=checkpoint_path)
    trainer.test(model=model, datamodule=datamodule)
    return
