import argparse
import logging
import os

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--train-path', type=str)
parser.add_argument('--data-dir', type=str, default='gs://aigagror/datasets')
parser.add_argument('--epochs', type=int)
parser.add_argument('--finetune-epoch', type=int)
parser.add_argument('--epoch-steps', type=int)
parser.add_argument('--bsz', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='*', default=[])
parser.add_argument('--log-level', choices=['debug', 'info', 'warn', 'error'], default='info')
parser.add_argument('--ds-ids', type=str, nargs='+')


def setup(args):
    # Paths
    args.model_path = os.path.join(args.train_path, 'model')
    args.downstream_path = os.path.join(args.train_path, 'downstream-tasks')

    # Remove any previous work
    os.system(f'gsutil -m rm -r {args.downstream_path}')

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
