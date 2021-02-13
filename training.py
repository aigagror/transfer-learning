import logging
import os
from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from data import load_ds, class_supervise, postprocess


def lr_scheduler(epoch, lr, decays):
    # Epoch is 0-indexed
    if epoch in decays:
        lr *= 0.1
    return lr


def extract_features(ds, feat_model):
    # Get the corresponding features for each image
    feats = tf.data.Dataset.from_tensor_slices(feat_model.predict(ds.batch(1024)))

    # Create feature, label dataset
    labels = ds.map(lambda x, y: y)
    ds_feats = tf.data.Dataset.zip((feats, labels))

    return ds_feats


def class_transfer_learn(args, strategy, feat_model, ds_id):
    # Load dataset
    ds_train, ds_val, info = load_ds(args, ds_id)
    nclass = info.features['label'].num_classes
    train_size = info.splits['train'].num_examples
    logging.info(f'{ds_id}, {nclass} classes, {train_size} train examples')

    # Map to classification format
    ds_class_train, ds_class_val = ds_train.map(class_supervise), ds_val.map(class_supervise)

    # Map images to features
    ds_feat_train = extract_features(ds_class_train, feat_model)
    ds_feat_val = extract_features(ds_class_val, feat_model)

    # Postprocess
    ds_class_train = postprocess(args, ds_class_train)
    ds_class_val = postprocess(args, ds_class_val)
    ds_feat_train = postprocess(args, ds_feat_train)
    ds_feat_val = postprocess(args, ds_feat_val)

    # Make classifier
    with strategy.scope():
        classifier = tf.keras.Sequential([
            tf.keras.Input([2048]),
            tf.keras.layers.Dense(nclass)
        ])
        optimizer = tfa.optimizers.LAMB(args.lr, weight_decay_rate=args.weight_decay)
        classifier.compile(optimizer, loss='sparse_categorical_crossentropy',
                           metrics='acc', steps_per_execution=100)

    # Train the classifier
    callbacks = [
        tf.keras.callbacks.TensorBoard(os.path.join(args.downstream_path, ds_id),
                                       write_graph=False, profile_batch=0),
        tf.keras.callbacks.LearningRateScheduler(
            partial(lr_scheduler, decays=args.lr_decays)
        )
    ]
    classifier.fit(ds_feat_train.repeat(), validation_data=ds_feat_val,
                   epochs=args.finetune_epoch or args.epochs,
                   steps_per_epoch=args.epoch_steps,
                   callbacks=callbacks)
    classifier_metrics = classifier.evaluate(ds_feat_val)

    # Save the classifer
    classifier.save(os.path.join(args.downstream_path, 'classifier'))

    # Create the transfer model
    with strategy.scope():
        clone_feat_model = tf.keras.models.clone_model(feat_model)
        clone_feat_model.trainable = True
        output = classifier(clone_feat_model.output)
        transfer_model = tf.keras.Model(clone_feat_model.input, output)
        transfer_model.compile(loss='sparse_categorical_crossentropy',
                               metrics='acc', steps_per_execution=50)

    # Verify the accuracy of the transfer model
    transfer_metrics = transfer_model.evaluate(ds_class_val)
    tf.debugging.assert_near(transfer_metrics, classifier_metrics)

    # Train the whole transfer model
    transfer_model.fit(ds_class_train.repeat(), validation_data=ds_class_val,
                       initial_epoch=args.finetune_epoch or args.epochs, epochs=args.epochs,
                       steps_per_epoch=args.epoch_steps,
                       callbacks=callbacks)

    # Save the transfer model
    transfer_model.save(os.path.join(args.downstream_path, 'transfer_model'))
