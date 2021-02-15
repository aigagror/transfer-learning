import logging
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data import load_ds, postprocess
from models import load_feat_model


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
    return feat_ds


def class_transfer_learn(args, strategy, ds_id):
    # Load dataset
    ds_train, info = load_ds(args, ds_id, 'train')
    ds_val = None
    for split in ['test', 'validation']:
        if split in info.splits:
            ds_val = load_ds(args, ds_id, split)
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
        optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
    classifier.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)

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
    ds_feat_train, ds_feat_val = extract_features(ds_train, feat_model), extract_features(ds_val, feat_model)
    classifier.fit(postprocess(ds_feat_train, args.linear_bsz, repeat=True),
                   validation_data=postprocess(ds_feat_val, args.linear_bsz),
                   epochs=args.finetune_epoch or args.epochs,
                   steps_per_epoch=args.epoch_steps,
                   callbacks=callbacks)

    # Verify classifer metrics match with
    logging.info('verifying metrics between classifier and transfer model')
    classifier_metrics = classifier.evaluate(ds_feat_val.batch(1024))
    transfer_metrics = transfer_model.evaluate(ds_val.batch(1024))
    tf.debugging.assert_near(classifier_metrics, transfer_metrics)

    # Train the whole transfer model
    ds_aug_train, info = load_ds(args, ds_id, 'train', augment=True)
    logging.info('fine-tuning whole model')
    transfer_model.trainable = True
    with strategy.scope():
        optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
        transfer_model.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)

    transfer_model.fit(postprocess(ds_aug_train, args.fine_bsz, repeat=True),
                       validation_data=postprocess(ds_val, args.fine_bsz),
                       initial_epoch=args.finetune_epoch or args.epochs, epochs=args.epochs,
                       steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Save the transfer model
    transfer_model.save(os.path.join(task_path, 'transfer_model'))
