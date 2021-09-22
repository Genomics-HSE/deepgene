import gin
from models import rm, data_gen_np, data, train_test
from pytorch_lightning import Trainer
from transformers import ReformerConfig

Trainer = gin.external_configurable(Trainer)
ReformerConfig = gin.external_configurable(ReformerConfig)

ReformerLabeler = gin.external_configurable(rm.ReformerLabeler)
Predictor = gin.external_configurable(rm.Predictor)
OrdinalHead = gin.external_configurable(rm.OrdinalHead)
DatasetPL = gin.external_configurable(data.DatasetPL)
DatasetTorch = gin.external_configurable(data.DatasetTorch)
get_liner_generator = gin.external_configurable(data_gen_np.get_liner_generator)
train_model = gin.external_configurable(train_test.train_model)
test_model = gin.external_configurable(train_test.test_model)
