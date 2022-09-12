import comet_ml
import time
import signal
import gin
from deepgene.utils import fit_model, test_model
import argparse


def handler(signum, frame):
    print('Received signal to end running', signum)
    raise KeyboardInterrupt


signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--gin_param", action="append")

    subparsers = parser.add_subparsers(dest="action")  # this line changed
    train_parser = subparsers.add_parser('fit')
    test_parser = subparsers.add_parser('test')

    args = parser.parse_args()
    print(args)
    gin.parse_config_files_and_bindings([args.model, args.data, args.train], args.gin_param)
    #print(gin.config._CONFIG)

    if args.action == "fit":
        fit_model(configs=gin.config._CONFIG)
    elif args.action == "test":
        test_model()
    else:
        raise ValueError("fit or predict")

