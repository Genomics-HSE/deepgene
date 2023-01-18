import time
import signal
import gin
import argparse

import pytorch_lightning as pl


import argparse

@gin.configurable
def kotok(n_class):
    print(n_class)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--gin_param", action="append")

    subparsers = parser.add_subparsers(dest="action")  # this line changed
    foo_parser = subparsers.add_parser('train')
    foo_parser.add_argument('-c', '--count')

    args = parser.parse_args()
    print(args.model)
    gin.parse_config_files_and_bindings([args.model, args.data, args.train], args.gin_param)
    print(gin.config._CONFIG)

    kotok()
