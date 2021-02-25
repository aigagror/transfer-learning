import contextlib
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.linear_model import LogisticRegression

from data import load_ds, postprocess
from models import load_feat_model


def lr_scheduler(epoch, lr, args):
    # Epoch is 0-indexed
    if epoch in args.lr_decays:
        return lr * 0.1
    return lr


@contextlib.contextmanager
def timed_execution(fn_name):
    t0 = time.time()
    yield
    dt = time.time() - t0
    logging.info(f'{fn_name} took {dt:.3} seconds')


def extract_features(class_ds, model):
    features = model.predict(class_ds.batch(1024))
    features = tf.data.Dataset.from_tensor_slices(features)
    labels = class_ds.map(lambda x, y: y)
    feat_ds = tf.data.Dataset.zip((features, labels))
    feat_ds = feat_ds.cache()
    return feat_ds


def get_optimizer(optimizer, lr, weight_decay):
    logging.debug(f'{lr} lr, {weight_decay} weight_decay')
    if optimizer == 'lamb':
        optimizer = tfa.optimizers.LAMB(lr, weight_decay_rate=weight_decay)
    elif optimizer == 'adamw':
        optimizer = tfa.optimizers.AdamW(weight_decay, lr)
    elif optimizer == 'sgdw':
        optimizer = tfa.optimizers.SGDW(weight_decay, lr, momentum=0.9, nesterov=True)
    else:
        raise Exception(f'unknown optimizer: {optimizer}')
    return optimizer


def class_transfer_learn(args, strategy, ds_id):
    verbosity = int(args.log_level == 'DEBUG')

    # Load dataset
    ds_train_no_augment, info = load_ds(args, ds_id, 'train')
    ds_val = None
    for split in ['test', 'validation']:
        if split in info.splits:
            ds_val, _ = load_ds(args, ds_id, split)
            break
    ds_val = ds_val or ds_train_no_augment

    nclass, train_size = info.features['label'].num_classes, info.splits['train'].num_examples
    logging.info(f'{ds_id}, {nclass} classes, {train_size} train examples')

    # Make transfer model
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with strategy.scope():
        input = tf.keras.Input([224, 224, 3])
        feat_model = load_feat_model(args, trainable=False)
        classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(nclass)
        ])
        output = classifier(feat_model(input, training=False))
        transfer_model = tf.keras.Model(input, output)

    logging.info(f'{len(transfer_model.losses)} regularization losses')
    if args.log_level == 'DEBUG':
        transfer_model.summary()

    # Extract features
    ds_feat_train, ds_feat_val = extract_features(ds_train_no_augment, feat_model), extract_features(ds_val, feat_model)

    train_metrics, val_metrics = [], []

    # Train classifier
    if args.linear_opt == 'lbfgs':
        logging.info('training classifier with LBFGS')
        train_feats, train_labels = zip(*ds_feat_train.batch(1024).as_numpy_iterator())
        train_feats, train_labels = np.concatenate(train_feats, axis=0), np.concatenate(train_labels, axis=0)
        logging.info(f'feats: {tf.reduce_min(train_feats):.3} min, {tf.reduce_max(train_feats):.3} max')

        with strategy.scope():
            classifier.compile(loss=ce_loss, metrics='acc', steps_per_execution=100)

        all_Cs = -np.array(args.log_wds)
        all_regs = np.logspace(min(all_Cs), max(all_Cs), num=args.grid_points)
        lbfgs = LogisticRegression(warm_start=True, multi_class='multinomial', n_jobs=-1)
        for unscaled_c in all_regs:
            c = unscaled_c / len(train_feats)
            logging.info(f'{unscaled_c:.3} unscaled C, {c:.3} scaled C')
            lbfgs.set_params(C=c)
            with timed_execution('LBFGS'):
                result = lbfgs.fit(train_feats, train_labels)
            classifier.layers[0].kernel.assign(result.coef_.T)
            classifier.layers[0].bias.assign(result.intercept_)

            train_metrics.append(classifier.evaluate(postprocess(ds_feat_train, 1024)))
            val_metrics.append(classifier.evaluate(postprocess(ds_feat_val, 1024)))
    else:
        logging.info('training classifier with gradient descent')
        all_regs = np.logspace(max(args.log_wds), min(args.log_wds), num=args.grid_points)
        with timed_execution(ds_id):
            for weight_decay in all_regs:
                with strategy.scope():
                    optimizer = get_optimizer(args.linear_opt, args.linear_lr, weight_decay)
                    classifier.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=100)
                classifier.fit(postprocess(ds_feat_train, args.linear_bsz, repeat=True),
                               initial_epoch=0, epochs=args.linear_epochs,
                               steps_per_epoch=args.epoch_steps, verbose=verbosity)
                train_metrics.append(classifier.evaluate(postprocess(ds_feat_train, 1024), verbose=verbosity))
                val_metrics.append(classifier.evaluate(postprocess(ds_feat_val, 1024), verbose=verbosity))

    # Plot metrics over regularization
    train_metrics, val_metrics = np.array(train_metrics), np.array(val_metrics)
    f, ax = plt.subplots(1, 2)
    f.set_size_inches(6, 3)
    log_cs = np.log10(all_regs)
    ax[0].set_xlabel('log reg'), ax[1].set_xlabel('log reg')

    ax[0].set_title('cross entropy')
    ax[0].plot(log_cs, train_metrics[:, 0], label='train')
    ax[0].plot(log_cs, val_metrics[:, 0], label='val')

    ax[1].set_title('accuracy')
    ax[1].plot(log_cs, train_metrics[:, 1], label='train')
    ax[1].plot(log_cs, val_metrics[:, 1], label='val')

    ax[0].legend(), ax[1].legend()
    plt.show()

    logging.info(f'{ds_id} -> {max(val_metrics[:, 1]) * 100:.3}%, {log_cs[np.argmax(val_metrics[:, 1])]:.3} log wd')

    # Compile the transfer model
    logging.info('fine-tuning whole model')
    if args.fine_epochs > 0:
        transfer_model.trainable = True
        with strategy.scope():
            optimizer = get_optimizer(args.fine_opt, args.fine_lr, args.fine_wd)
            transfer_model.compile(optimizer, loss=ce_loss, metrics='acc', steps_per_execution=200)

        # Finetune the transfer model
        ds_train, _ = load_ds(args, ds_id, 'train', augment=True)
        transfer_model.fit(postprocess(ds_train, args.fine_bsz, repeat=True),
                           validation_data=postprocess(ds_val, args.fine_bsz),
                           initial_epoch=args.linear_epochs, epochs=args.linear_epochs + args.fine_epochs,
                           steps_per_epoch=args.epoch_steps, verbose=verbosity)
