import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import deepgen.models
from deepgen.models import GruLabeler
from deepgen.data import DatasetXY


from pytorch_lightning.utilities.cli import MODEL_REGISTRY

MODEL_REGISTRY.register_classes(deepgen.models, pl.LightningModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")



if __name__ == '__main__':

    cli = LightningCLI(pl.LightningModule, pl.LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)


