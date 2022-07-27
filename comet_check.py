import comet_ml
import time
import signal
import gin
import argparse
from deepgen.utils import train_model, test_model



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
    print(gin.config.c)
