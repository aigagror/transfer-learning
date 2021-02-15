import logging
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data import load_ds, class_supervise, postprocess
from models import load_feat_model


def lr_scheduler(epoch, lr, args):
    # Epoch is 0-indexed
    curr_lr = args.lr
    for e in range(1, epoch + 1):
        if e in args.lr_decays:
            curr_lr *= 0.1
    return curr_lr


def class_transfer_learn(args, strategy, ds_id):
    # Load dataset
    ds_train, ds_val, info = load_ds(args, ds_id)
    nclass, train_size = info.features['label'].num_classes, info.splits['train'].num_examples
    logging.info(f'{ds_id}, {nclass} classes, {train_size} train examples')

    # Map to classification format
    ds_class_train, ds_class_val = ds_train.map(class_supervise), ds_val.map(class_supervise)

    # Postprocess
    ds_class_train, ds_class_val = postprocess(args, ds_class_train), postprocess(args, ds_class_val)

    # Make transfer model
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with strategy.scope():
        feat_model = load_feat_model(args, trainable=False)
        output = tf.keras.layers.Dense(nclass)(feat_model.output)
        transfer_model = tf.keras.Model(feat_model.input, output)
        optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
    transfer_model.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)

    if args.log_level == 'DEBUG':
        transfer_model.summary()

    # Train the classifier
    logging.info('training classifier')
    task_path = os.path.join(args.downstream_path, ds_id)
    callbacks = [
        tf.keras.callbacks.TensorBoard(task_path, write_graph=False, profile_batch=0),
        tf.keras.callbacks.LearningRateScheduler(
            partial(lr_scheduler, args=args),
            verbose=1 if args.log_level == 'DEBUG' else 0
        )
    ]
    transfer_model.fit(ds_class_train.repeat(), validation_data=ds_class_val,
                       epochs=args.finetune_epoch or args.epochs,
                       steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Train the whole transfer model
    logging.info('fine-tuning whole model')
    transfer_model.trainable = True
    transfer_model.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)
    transfer_model.fit(ds_class_train.repeat(), validation_data=ds_class_val,
                       initial_epoch=args.finetune_epoch or args.epochs, epochs=args.epochs,
                       steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Save the transfer model
    transfer_model.save(os.path.join(task_path, 'transfer_model'))
