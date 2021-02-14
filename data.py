from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds


def sample_bbox_crop(inputs):
    image = inputs['image']
    if 'bbox' in inputs:
        bbox = inputs['bbox']
    else:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0])
    bbox = tf.reshape(bbox, [1, 1, 4])

    distorted_bbox = tf.image.sample_distorted_bounding_box(tf.shape(image), bbox, min_object_covered=0.75)
    bbox_begin, bbox_size, _ = distorted_bbox

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)

    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                          target_height, target_width)
    inputs['image'] = image

    # Crop segmentation mask as well?
    if 'segmentation_mask' in inputs:
        mask = inputs['segmentation_mask']
        mask = tf.image.crop_to_bounding_box(mask, offset_y, offset_x,
                                             target_height, target_width)
        inputs['segmentation_mask'] = mask

    return inputs


def center_crop(inputs):
    image = inputs['image']
    shape = tf.shape(image)
    image_height, image_width = shape[0], shape[1]

    center_crop_size = tf.minimum(image_height, image_width)

    offset_height = ((image_height - center_crop_size) + 1) // 2
    offset_width = ((image_width - center_crop_size) + 1) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                          center_crop_size, center_crop_size)
    inputs['image'] = image
    return inputs


def apply_img_fn(inputs, img_fn):
    for key in ['image', 'segmentation_mask']:
        if key in inputs:
            inputs[key] = img_fn(inputs[key])
    return inputs


def as_supervised(inputs, target):
    return inputs['image'], inputs[target]


class_supervise = partial(as_supervised, target='label')

segment_supervise = partial(as_supervised, target='segmentation_mask')


def preprocess(ds, training):
    if training:
        ds = ds.map(sample_bbox_crop, tf.data.AUTOTUNE)
    else:
        ds = ds.map(center_crop, tf.data.AUTOTUNE)

    to_uint8 = partial(tf.cast, dtype=tf.uint8)
    resize_224 = partial(tf.image.resize, size=[224, 224])
    img_fn = lambda x: to_uint8(resize_224(x))

    img_preprocess = partial(apply_img_fn, img_fn=img_fn)

    ds = ds.map(img_preprocess, tf.data.AUTOTUNE)
    return ds


def load_ds(args, ds_id):
    ds_train, info = tfds.load(ds_id, split='train', data_dir=args.data_dir, try_gcs=True, shuffle_files=False,
                               with_info=True)
    ds_val = None
    for split in ['test', 'validation']:
        if split in info.splits:
            ds_val = tfds.load(ds_id, split=split, data_dir=args.data_dir, try_gcs=True)
            break
    ds_val = ds_val or ds_train

    # Preprocess
    processed_ds_train = preprocess(ds_train, training=True)
    processed_ds_val = preprocess(ds_val, training=False)

    # Show examples if debug level is log
    if args.log_level == 'DEBUG':
        for image_key in ['image', 'segmentation_mask']:
            if image_key in info.features:
                tfds.show_examples(processed_ds_val, info, image_key=image_key, rows=1)
                tfds.show_examples(processed_ds_train, info, image_key=image_key, rows=1)

    return processed_ds_train, processed_ds_val, info


def postprocess(args, ds):
    return (ds
            .cache()
            .shuffle(len(ds))
            .batch(args.bsz)
            .prefetch(tf.data.AUTOTUNE)
            )
