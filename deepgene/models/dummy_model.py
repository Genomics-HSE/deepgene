from typing import Union, Tuple, Any, Dict

from pytorch_lightning import LightningModule
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


patch_typeguard()


class DummyModel(LightningModule):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.loss = lambda x, y: 1

    @typechecked
    def training_step(self, batch: Tuple[TensorType["batch", "genome_length"], TensorType["batch", "genome_length"]],
                      batch_ix: Any) -> Union[TensorType[...], Dict[str, Any]]:
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        loss = self.loss1(logits, y_batch)
        return loss

    @typechecked
    def validation_step(self, batch: Tuple[TensorType["batch", "genome_length"], TensorType["batch", "genome_length"]],
                        batch_ix: Any) -> Union[TensorType[...], Dict[str, Any]]:
        x_batch, y_batch = batch
        logits = self.forward(x_batch)
        loss = self.loss(logits, y_batch)
        return loss

    @typechecked
    def test_step(self, batch: Tuple[TensorType["batch", "genome_length"], TensorType["batch", "genome_length"]],
                  batch_idx: Any) -> TensorType["batch", "genome_length", "hidden_size"]:
        X_batch, y_batch = batch
        preds = self.forward(X_batch)
        return preds
