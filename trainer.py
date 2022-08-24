import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")



if __name__ == '__main__':

    cli = LightningCLI(pl.LightningModule, pl.LightningDataModule,
                       subclass_mode_model=True,
                       subclass_mode_data=True,
                       save_config_overwrite=True)


