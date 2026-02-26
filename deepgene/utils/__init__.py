import gin
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import CometLogger
from .train_test import fit_model, test_model
from .viz import viz_heatmap, create_heatmap, create_dist_plot, create_NE_plot
from .cli import CLI
from .add_utils import from_probability_to_N, from_N_to_probability

Trainer = gin.external_configurable(Trainer)
CometLogger = gin.external_configurable(CometLogger)
ModelCheckpoint = gin.external_configurable(ModelCheckpoint)
EarlyStopping = gin.external_configurable(EarlyStopping)

fit_model = gin.external_configurable(fit_model)
test_model = gin.external_configurable(test_model)
