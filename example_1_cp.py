import pytorch_lightning as pl
from deepgene.models import DeepCompressor
from deepgene.data import DatasetPL, get_liner_generator

if __name__ == '__main__':
    trainer = pl.Trainer(
        default_root_dir="output",
        max_epochs=1,
        checkpoint_callback=False,
        logger=False,
        gpus=0,
    )
    
    model = DeepCompressor(
        channel_size=32,
        conv_kernel_size=5,
        conv_stride=1,
        num_layers=2,
        dropout_p=0.1,
        pool_kernel_size=2,
        n_output=1,
        seq_len=3000,
    )
    
    train_generator = get_liner_generator(
        num_genomes=20,
        genome_length=3000,
        num_generators=1
    )
    
    data = DatasetPL(
        train_generator=train_generator,
        val_generator=None,
        test_generator=None,
        batch_size=2,
        num_workers=4,
    )

    trainer.fit(model=model, datamodule=data)

    model.save(trainer, "output/model-weights.ckpt")

