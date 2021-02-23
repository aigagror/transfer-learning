import argparse
import logging
import os
import shutil

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--ds-ids', type=str, nargs='+')
parser.add_argument('--train-path', type=str)
parser.add_argument('--data-dir', type=str, default='gs://aigagror/datasets')

parser.add_argument('--optimizer', choices=['lamb', 'lbfgs', 'sgdw', 'adamw'])

parser.add_argument('--epochs', type=int)
parser.add_argument('--finetune-epoch', type=int)
parser.add_argument('--epoch-steps', type=int)

parser.add_argument('--linear-bsz', type=int)
parser.add_argument('--fine-bsz', type=int)

parser.add_argument('--linear-wd', type=float)
parser.add_argument('--fine-wd', type=float)

parser.add_argument('--linear-lr', type=float)
parser.add_argument('--fine-lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='*', default=[])

parser.add_argument('--log-level', choices=['debug', 'info', 'warn', 'error'], default='info')


def setup(args):
    # Paths
    args.model_path = os.path.join(args.train_path, 'model')
    args.downstream_path = os.path.join(args.train_path, 'downstream-tasks')

    # Remove any previous work
    logging.info('clearing remote logs')
    os.system(f'gsutil -m rm -r {args.downstream_path}')

    logging.info('clearing local logs')
    shutil.rmtree('./logs', ignore_errors=True)

    # Logs
    args.log_level = args.log_level.upper()
    logging.getLogger().setLevel(args.log_level)
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(args.log_level)
    for h in tf_logger.handlers:
        tf_logger.removeHandler(h)

    # TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    return strategy
