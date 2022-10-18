import sys
from os.path import join

import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.loggers.comet import CometLogger
from tqdm import tqdm
import transformers
from typing import Tuple


def fit_model(trainer: Trainer,
              model: LightningModule,
              data_module: LightningDataModule,
              output: str,
              model_name: str,
              configs: dict,
              config_files: Tuple[str]
              ):

    print("Running {}-model...".format(model.name))
    trainer.logger.experiment.log_code(config_files[0])
    trainer.logger.experiment.log_code(config_files[1])
    trainer.logger.experiment.log_code(config_files[2])

    trainer.fit(model=model, datamodule=data_module)

    checkpoint_path = join(output, model_name)
    print("Saving the model to ", checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    print(trainer.checkpoint_callback.best_model_path)

    if isinstance(trainer.logger, CometLogger):
        trainer.logger.experiment.log_model(name="best_model",
                                            file_or_folder=trainer.checkpoint_callback.best_model_path)

    return trainer, model


def test_model(trainer: Trainer,
               model: LightningModule,
               checkpoint_path: str,
               test_output: str,
               model_name: str,
               datamodule: LightningDataModule,
               ):
    checkpoint_path = join(test_output, model_name)
    predict_by_model(model, checkpoint_path, test_output, datamodule)
    return model


def predict_by_model(model: LightningModule,
                     checkpoint_path: str,
                     test_output: str,
                     datamodule: LightningDataModule,
                     ):
    print("Checkpoint path is", checkpoint_path)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path).eval()

    with torch.no_grad():
        for i, (x, y_true) in tqdm(enumerate(datamodule.test_dataloader())):
            y_pred = model(x)
            torch.save(x, join(test_output, str(i) + "_x.pt"))
            torch.save(y_true, join(test_output, str(i) + "_y_true.pt"))
            torch.save(y_pred, join(test_output, str(i) + "_y_pred.pt"))

    return
