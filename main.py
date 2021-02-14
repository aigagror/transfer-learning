import logging
import os
import shutil

from models import load_feat_model
from training import class_transfer_learn
from utils import setup


def run(args):
    strategy = setup(args)

    feat_model = load_feat_model(args, strategy)

    for ds_id in args.ds_ids:
        class_transfer_learn(args, strategy, feat_model, ds_id)

    logging.info('clearing local log folder')
    shutil.rmtree('./logs', ignore_errors=True)

    logging.info('downloading GCP logs')
    os.system(f'gsutil -m cp -r {args.downstream_path} ./')
    base_dir = os.path.basename(args.downstream_path)
    shutil.move(base_dir, 'logs')
    logging.info('GCP logs downloaded')

    tensorboard_cmd = "tensorboard dev upload " \
                      "--logdir logs " \
                      f"--name '{args.train_path}' " \
                      f"--description '{args}' " \
                      "--one_shot"
    return tensorboard_cmd
