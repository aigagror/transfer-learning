import logging

import tensorflow as tf


def load_feat_model(args, trainable):
    logging.info('loading base model')
    base_model = tf.keras.models.load_model(args.model_path, compile=False)
    base_model.trainable = trainable

    logging.info('extracting feature model')
    try:
        input = base_model.get_layer('image').input
    except:
        # Backwards compatability
        input = base_model.get_layer('imgs').input

    try:
        feat_out = base_model.get_layer('feature').output
    except:
        # Backwards compatability
        feat_out = base_model.get_layer('feats').output

    feat_model = tf.keras.Model(input, feat_out)

    if args.log_level == 'DEBUG':
        base_model.summary()
        feat_model.summary()

    return feat_model
