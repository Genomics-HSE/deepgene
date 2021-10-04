import gin
import argparse
from models import train_model, test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Genomics')
    parser.add_argument("--config", type=str, default="")
    action_parsers = parser.add_subparsers(title='actions', dest='action')
    train_parser = action_parsers.add_parser('train')
    
    predict_parser = action_parsers.add_parser('test')
    predict_parser.add_argument('--path', type=str)
    args = parser.parse_args()
    
    gin.parse_config_file(args.config)
    
    if args.action == "train":
        train_model()
    elif args.action == "test":
        test_model()
    else:
        ValueError("Choose train or predict")
