import gin
import torch.nn as nn
from transformers import ReformerConfig

from .compress import DeepCompressor
from .gru import GruLabeler, GruLabelerOrdinal
from .reformer import ReformerLabeler, ReformerPreTrainerLM
from .simple import SimpleLabeler, WindowSlider
from .base_models import Predictor, OrdinalHead

ReformerConfig = gin.external_configurable(ReformerConfig)

NoEmbedding = gin.external_configurable(base_models.NoEmbedding)
Embedding = gin.external_configurable(nn.Embedding)
ConvEmbedding = gin.external_configurable(base_models.ConvEmbedding)

SimpleLabeler = gin.external_configurable(SimpleLabeler)
WindowSlider = gin.external_configurable(WindowSlider)

ReformerLabeler = gin.external_configurable(ReformerLabeler)
ReformerPreTrainerLM = gin.external_configurable(ReformerPreTrainerLM)

GruLabeler = gin.external_configurable(GruLabeler)
GruLabelerOrdinal = gin.external_configurable(GruLabelerOrdinal)
Predictor = gin.external_configurable(Predictor)
OrdinalHead = gin.external_configurable(OrdinalHead)

DeepCompressor = gin.external_configurable(DeepCompressor)




###### SPIDNA network
from .spidna import SPIDNA, SPIDNA2_adaptive