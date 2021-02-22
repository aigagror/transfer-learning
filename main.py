import logging
import os
import shutil

from training import class_transfer_learn
from utils import setup


def run(args):
    # Setup
    strategy = setup(args)
    logging.info(args)

    # Transfer learn
    for ds_id in args.ds_ids:
        class_transfer_learn(args, strategy, ds_id)

    # Get logs
    logging.info('downloading GCP logs')
    os.system(f'gsutil -m cp -r {args.downstream_path} ./')
    base_dir = os.path.basename(args.downstream_path)
    shutil.move(base_dir, 'logs')
    logging.info('GCP logs downloaded')

    # Return tensorboard command
    tensorboard_cmd = "tensorboard dev upload " \
                      "--logdir logs " \
                      f"--name '{args.train_path}' " \
                      f"--description '{args.linear_wd} linear l2, {args.fine_wd} fine l2' " \
                      "--one_shot"
    return tensorboard_cmd
