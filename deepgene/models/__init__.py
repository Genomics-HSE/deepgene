import gin
import torch.nn as nn
try:
    import pytorch_lightning as pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pl = None  # type: ignore
try:
    from transformers import ReformerConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ReformerConfig = None  # type: ignore[assignment]

try:  # pragma: no cover
    from pytorch_lightning.utilities.cli import MODEL_REGISTRY  # type: ignore
except Exception:  # noqa: BLE001
    MODEL_REGISTRY = None  # type: ignore

from . import base_models
from .gru import GruLabeler, GruLabelerOrdinal
from .gru_com_dist import GruComDistLabeler
from .lstm import LstmLabeler
from .simple import SimpleLabeler, WindowSlider
from .base_models import Predictor, OrdinalHead, CategoricalModel
from .dummy_model import DummyModel


if ReformerConfig is not None:  # pragma: no cover
    ReformerConfig = gin.external_configurable(ReformerConfig)

NoEmbedding = gin.external_configurable(base_models.NoEmbedding)
Embedding = gin.external_configurable(nn.Embedding)
ConvEmbedding = gin.external_configurable(base_models.ConvEmbedding)

SimpleLabeler = gin.external_configurable(SimpleLabeler)
WindowSlider = gin.external_configurable(WindowSlider)

# Reformer is optional: depends on `transformers`.
try:  # pragma: no cover
    from .reformer import ReformerLabeler, ReformerPreTrainerLM

    ReformerLabeler = gin.external_configurable(ReformerLabeler)
    ReformerPreTrainerLM = gin.external_configurable(ReformerPreTrainerLM)
except ModuleNotFoundError:
    # Allow importing deepgene.models without transformers installed (GRU/LSTM use-cases).
    pass
#
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