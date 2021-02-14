import logging

import tensorflow as tf


def load_feat_model(args, strategy):
    logging.info('loading base model')
    with strategy.scope():
        base_model = tf.keras.models.load_model(args.model_path, compile=False)
        base_model.trainable = False

    logging.info('extracting feature model')
    with strategy.scope():
        try:
            input = base_model.get_layer('image').input
        except:
            input = base_model.get_layer('imgs').input
        feat_out = base_model.get_layer('feats').output
        feat_model = tf.keras.Model(input, feat_out)

    if args.log_level == 'DEBUG':
        base_model.summary()
        feat_model.summary()

    return feat_model
