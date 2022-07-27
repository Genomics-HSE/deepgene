import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from deepgen.models import GruLabeler, Mock
from deepgen.data import DatasetXY
import deepgen.models

if __name__ == '__main__':
    cli = LightningCLI(auto_registry=True, subclass_mode_model=True, subclass_mode_data=True)