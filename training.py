import logging
import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from data import load_ds, postprocess
from models import load_feat_model
from sklearn.linear_model import LogisticRegression


def lr_scheduler(epoch, lr, args):
    # Epoch is 0-indexed
    curr_lr = args.lr
    for e in range(1, epoch + 1):
        if e in args.lr_decays:
            curr_lr *= 0.1
    return curr_lr


def extract_features(class_ds, model):
    features = model.predict(class_ds.batch(1024))
    features = tf.data.Dataset.from_tensor_slices(features)
    labels = class_ds.map(lambda x, y: y)
    feat_ds = tf.data.Dataset.zip((features, labels))
    feat_ds = feat_ds.cache()
    return feat_ds


def class_transfer_learn(args, strategy, ds_id):
    # Load dataset
    ds_train, info = load_ds(args, ds_id, 'train')
    ds_val = None
    for split in ['test', 'validation']:
        if split in info.splits:
            ds_val, _ = load_ds(args, ds_id, split)
            break
    ds_val = ds_val or ds_train

    nclass, train_size = info.features['label'].num_classes, info.splits['train'].num_examples
    logging.info(f'{ds_id}, {nclass} classes, {train_size} train examples')

    # Make transfer model
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with strategy.scope():
        feat_model = load_feat_model(args, trainable=False)
        classifier = tf.keras.Sequential([tf.keras.layers.Dense(nclass)])
        output = classifier(feat_model.output)
        transfer_model = tf.keras.Model(feat_model.input, output)

    logging.info(f'{len(transfer_model.losses)} regularization losses')
    if args.log_level == 'DEBUG':
        transfer_model.summary()

    # Extract features
    ds_feat_train, ds_feat_val = extract_features(ds_train, feat_model), extract_features(ds_val, feat_model)

    if args.optimizer == 'lbfgs':
        logging.info('training classifier with LBFGS')
        train_feats, train_labels = zip(*ds_feat_train.batch(1024).as_numpy_iterator())
        train_feats, train_labels = np.concatenate(train_feats, axis=0), np.concatenate(train_labels, axis=0)
        result = LogisticRegression(C=(1 / args.weight_decay)).fit(train_feats, train_labels)
        classifier.layers[0].kernel.assign(result.coef_.T)
        classifier.layers[0].bias.assign(result.intercept_)

        with strategy.scope():
            classifier.compile(loss=ce_loss, metrics='acc', steps_per_execution=100)
        classifier.evaluate(postprocess(ds_feat_val, args.linear_bsz))
    else:
        # Train the classifier with gradient descent
        task_path = os.path.join(args.downstream_path, ds_id)
        callbacks = [
            tf.keras.callbacks.TensorBoard(task_path, write_graph=False, profile_batch=0),
            tf.keras.callbacks.LearningRateScheduler(
                partial(lr_scheduler, args=args),
                verbose=1 if args.log_level == 'DEBUG' else 0
            )
        ]

        logging.info('training classifier with gradient descent')
        with strategy.scope():
            optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
            classifier.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)
        classifier.fit(postprocess(ds_feat_train, args.linear_bsz, repeat=True),
                       validation_data=postprocess(ds_feat_val, args.linear_bsz),
                       epochs=args.finetune_epoch or args.epochs, steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Compile the transfer model
    logging.info('fine-tuning whole model')
    transfer_model.trainable = True
    with strategy.scope():
        optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
        transfer_model.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)

    # Finetune the transfer model
    ds_train, _ = load_ds(args, ds_id, 'train', augment=True)
    transfer_model.fit(postprocess(ds_train, args.fine_bsz, repeat=True),
                       validation_data=postprocess(ds_val, args.fine_bsz),
                       initial_epoch=args.finetune_epoch or args.epochs, epochs=args.epochs,
                       steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Save the transfer model
    transfer_model.save(os.path.join(task_path, 'transfer_model'))
