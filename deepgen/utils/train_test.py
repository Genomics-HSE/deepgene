import sys
from os.path import join

import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from tqdm import tqdm
import transformers


def train_model(trainer: Trainer,
                model: LightningModule,
                data_module: LightningDataModule,
                output: str,
                model_name: str,
                resume: bool,
                configs: dict
                ):
    print("Running {}-model...".format(model.name))

    trainer.logger.experiment.log_parameters(configs)

    if resume:
        # load model
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.fit(model=model, datamodule=data_module)

    checkpoint_path = join(output, model_name)
    print("Saving the model to ", checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    print(trainer.checkpoint_callback.best_model_path)
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
