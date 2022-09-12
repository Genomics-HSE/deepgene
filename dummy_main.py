from pytorch_lightning import Trainer
from deepgene.models import DummyModel
from deepgene.data import DummyDataset
import numpy as np


if __name__ == '__main__':
    trainer = Trainer(
        max_steps=10,
        enable_checkpointing=True,
        auto_lr_find=True,
        default_root_dir=".",
        log_every_n_steps=1,
        val_check_interval=5,
        limit_val_batches=1,
    )
    model = DummyModel()
    datamodule = DummyDataset()

    trainer.fit(model, datamodule)


