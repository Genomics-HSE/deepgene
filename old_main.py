import comet_ml
import sys

import time
import signal
import gin
import argparse
from deepgene.utils import train_model, test_model


def handler(signum, frame):
    print('Received signal to end running', signum)
    raise KeyboardInterrupt


signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument("--config", type=str, default="")
    action_parsers = parser.add_subparsers(title='actions', dest='action')
    train_parser = action_parsers.add_parser('train')

    predict_parser = action_parsers.add_parser('test')
    predict_parser.add_argument('--path', type=str)
    args = parser.parse_args()

    print(args.config)
    gin.parse_config_file(args.config)

    print(gin.config._CONFIG)
    if args.action == "train":
        train_model(configs=gin.config._CONFIG)
    elif args.action == "test":
        test_model()
    else:
        ValueError("Choose train or predict")