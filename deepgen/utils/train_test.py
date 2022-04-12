import sys
from os.path import join

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
import transformers

from deepgen.models import ReformerLabeler


def train_model(trainer: Trainer,
                model: LightningModule,
                data_module: LightningDataModule,
                checkpoint_path: str,
                resume: bool
                ):
    print("Running {}-model...".format(model.name))
    
    if resume:
        # load model
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.fit(model=model, datamodule=data_module)
    print("Saving the model to ", checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    return trainer, model


def test_model(trainer: Trainer,
               model: LightningModule,
               checkpoint_path: str,
               test_output: str,
               datamodule: LightningDataModule,
               ):
    print(checkpoint_path)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # trainer.test(model=model, datamodule=datamodule)
    return model


def predict_by_model(trainer: Trainer,
               model: LightningModule,
               checkpoint_path: str,
               test_output: str,
               datamodule: LightningDataModule,
               ):
    print(checkpoint_path)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    trainer.test(model=model, datamodule=datamodule)
    return