import comet_ml
import time
import signal
import gin
from absl import flags, app
from deepgen.utils import train_model, test_model


def handler(signum, frame):
    print('Received signal to end running', signum)
    raise KeyboardInterrupt


signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

flags.DEFINE_string('data', None, 'Gin data')
flags.DEFINE_string('model', None, 'Gin model')
flags.DEFINE_string('train', None, 'Gin train')
flags.DEFINE_enum('action', 'fit', ['fit', 'test'], "Action to do")

flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

flags.mark_flag_as_required("data")
flags.mark_flag_as_required("model")
flags.mark_flag_as_required("train")


def main(argv):
    print(argv)
    print(FLAGS.data, FLAGS.model, FLAGS.train)

    gin.parse_config_files_and_bindings([FLAGS.data,
                                         FLAGS.model,
                                         FLAGS.train
                                         ], FLAGS.gin_param)
    print(gin.config._CONFIG)

    if FLAGS.action == "fit":
        train_model(configs=gin.config._CONFIG)
    elif FLAGS.action == "test":
        test_model()
    else:
        ValueError("Choose train or predict")


if __name__ == '__main__':
    app.run(main)