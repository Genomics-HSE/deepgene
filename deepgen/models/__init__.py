import gin
import torch.nn as nn
import pytorch_lightning as pl
from transformers import ReformerConfig

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from .gru import GruLabeler, GruLabelerOrdinal
from .gru_com_dist import GruComDistLabeler
from .lstm import LstmLabeler
from .reformer import ReformerLabeler, ReformerPreTrainerLM
from .simple import SimpleLabeler, WindowSlider
from .base_models import Predictor, OrdinalHead, CategoricalModel


ReformerConfig = gin.external_configurable(ReformerConfig)

NoEmbedding = gin.external_configurable(base_models.NoEmbedding)
Embedding = gin.external_configurable(nn.Embedding)
ConvEmbedding = gin.external_configurable(base_models.ConvEmbedding)

SimpleLabeler = gin.external_configurable(SimpleLabeler)
WindowSlider = gin.external_configurable(WindowSlider)

ReformerLabeler = gin.external_configurable(ReformerLabeler)
ReformerPreTrainerLM = gin.external_configurable(ReformerPreTrainerLM)

GruLabeler = gin.external_configurable(GruLabeler)
GruComDistLabeler = gin.external_configurable(GruComDistLabeler)
GruLabelerOrdinal = gin.external_configurable(GruLabelerOrdinal)
Predictor = gin.external_configurable(Predictor)
OrdinalHead = gin.external_configurable(OrdinalHead)


LstmLabeler = gin.external_configurable(LstmLabeler)
LSTM = gin.external_configurable(nn.LSTM)


###### SPIDNA network
from .spidna import SPIDNA, SPIDNA2_adaptive

if __name__ == '__main__':
    print(MODEL_REGISTRY)