import comet_ml
import time
import signal
import gin
from deepgene.utils import fit_model, test_model, CLI
import argparse


def handler(signum, frame):
    print('Received signal to end running', signum)
    raise KeyboardInterrupt


signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)



if __name__ == '__main__':

    args = CLI()

    if args.action == "fit":
        fit_model(configs=gin.config._CONFIG)
    elif args.action == "test":
        test_model()
    else:
        raise ValueError("fit or predict")

