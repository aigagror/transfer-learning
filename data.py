from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

import augmentations


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

    # Crop segmentation mask as well?
    if 'segmentation_mask' in inputs:
        mask = inputs['segmentation_mask']
        mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width,
                                             center_crop_size, center_crop_size)
        inputs['segmentation_mask'] = mask

    inputs['image'] = image
    return inputs


def apply_img_fn(inputs, img_fn):
    for key in ['image', 'segmentation_mask']:
        if key in inputs:
            inputs[key] = img_fn(inputs[key])
    return inputs


def flip_horizontal(inputs):
    for key in ['image', 'segmentation_mask']:
        if key in inputs:
            inputs[key] = tf.image.flip_left_right(inputs[key])
    return inputs


def rand_flip(inputs):
    return tf.cond(tf.random.uniform([]) > 0.5, lambda: inputs, lambda: flip_horizontal(inputs))


def autoaugment(inputs):
    inputs['image'] = augmentations.AutoAugment().distort(inputs['image'])
    return inputs


def as_supervised(inputs, target):
    return inputs['image'], inputs[target]


class_supervise = partial(as_supervised, target='label')
segment_supervise = partial(as_supervised, target='label')


def preprocess(ds, augment):
    ds = ds.cache()
    if augment:
        ds = ds.map(sample_bbox_crop, tf.data.AUTOTUNE)
        ds = ds.map(rand_flip, tf.data.AUTOTUNE)
    else:
        ds = ds.map(center_crop, tf.data.AUTOTUNE)

    to_uint8 = partial(tf.cast, dtype=tf.uint8)
    resize_224 = partial(tf.image.resize, size=[224, 224], method='bicubic')
    img_fn = lambda x: to_uint8(resize_224(x))

    img_preprocess = partial(apply_img_fn, img_fn=img_fn)

    ds = ds.map(img_preprocess, tf.data.AUTOTUNE)

    if not augment:
        # Non augmented images are always the same so we can cache the final images.
        ds = ds.cache()

    return ds


def load_ds(args, ds_id, split, augment=False):
    _, info = tfds.load(ds_id, data_dir=args.data_dir, try_gcs=True, with_info=True)
    ds = tfds.load(ds_id, split=split, data_dir=args.data_dir, try_gcs=True)

    # Preprocess
    processed_ds = preprocess(ds, augment)

    # Show examples if debug level is log
    if args.log_level == 'DEBUG':
        for image_key in ['image', 'segmentation_mask']:
            if image_key in info.features:
                tfds.show_examples(processed_ds, info, image_key=image_key, rows=1, cols=5)

    class_ds = processed_ds.map(class_supervise, tf.data.AUTOTUNE)

    return class_ds, info


def postprocess(ds, bsz, repeat=False):
    ds = ds.shuffle(len(ds))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(bsz)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
