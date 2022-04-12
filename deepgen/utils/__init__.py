import gin
from pytorch_lightning import Trainer
from .train_test import train_model, test_model
from .viz import viz_heatmap

Trainer = gin.external_configurable(Trainer)

train_model = gin.external_configurable(train_model)
test_model = gin.external_configurable(test_model)
