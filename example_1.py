import time
import signal
import gin
import argparse
from models import train_model, test_model, DeepCompressor

import pytorch_lightning as pl


if __name__ == '__main__':
    trainer = pl.Trainer()
    model = 