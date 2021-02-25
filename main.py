import logging

from training import class_transfer_learn
from utils import setup


def run(args):
    # Setup
    strategy = setup(args)
    logging.info(args)

    # Transfer learn
    for ds_id in args.ds_ids:
        class_transfer_learn(args, strategy, ds_id)
