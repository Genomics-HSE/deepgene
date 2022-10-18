import argparse
import gin


def CLI():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--gin_param", action="append")

    subparsers = parser.add_subparsers(dest="action")  # this line changed
    train_parser = subparsers.add_parser('fit')
    test_parser = subparsers.add_parser('test')
    val_parser = subparsers.add_parser('validate')
    predict_parser = subparsers.add_parser('predict')

    args = parser.parse_args()
    print(args)
    gin.parse_config_files_and_bindings([args.model, args.data, args.train], args.gin_param)
    #print(gin.config._CONFIG)

    return args

