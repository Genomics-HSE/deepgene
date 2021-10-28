import gin
import torch.nn as nn
from models import reformer, data_gen_np, data, train_test, gru, base_models, simple
from pytorch_lightning import Trainer
from transformers import ReformerConfig

Trainer = gin.external_configurable(Trainer)
ReformerConfig = gin.external_configurable(ReformerConfig)

NoEmbedding = gin.external_configurable(base_models.NoEmbedding)
Embedding = gin.external_configurable(nn.Embedding)
ConvEmbedding = gin.external_configurable(base_models.ConvEmbedding)

SimpleLabeler = gin.external_configurable(simple.SimpleLabeler)
WindowSlider = gin.external_configurable(simple.WindowSlider)

ReformerLabeler = gin.external_configurable(reformer.ReformerLabeler)
GruLabeler = gin.external_configurable(gru.GruLabeler)
GruLabelerOrdinal = gin.external_configurable(gru.GruLabelerOrdinal)
Predictor = gin.external_configurable(base_models.Predictor)
OrdinalHead = gin.external_configurable(base_models.OrdinalHead)

DatasetPL = gin.external_configurable(data.DatasetPL)
DatasetTorch = gin.external_configurable(data.DatasetTorch)
get_liner_generator = gin.external_configurable(data_gen_np.get_liner_generator)
train_model = gin.external_configurable(train_test.train_model)
test_model = gin.external_configurable(train_test.test_model)
