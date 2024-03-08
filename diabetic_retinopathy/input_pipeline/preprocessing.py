'''
This script was developed by Shuaike Liu to preprocess the pictures and add augmentation methods.
It includes two functions preprocess() and augment(), both of which are called in the datasets.py script.

The function preprocess does preprocessing to image including normalizing and resizing.
It take image and its label, as well as the wished size for resizing as input. And it returns the image with label after preprocessing.

The function augment includes 6 different augmentation methods.
Besides of image and its label, the parameters also includs apply_flip, x_bright, x_contrast, x_saturation, x_quality,
which can be configured while calling to define the strength of 6 different augmentaion methods

'''
import tensorflow as tf


def preprocess(image, label, img_height=255, img_width=255):
    """Dataset preprocessing: Normalizing and resizing"""

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.0

    # Resize image
    # image = tf.image.resize(image, size=(img_height, img_width))

    # Expand label dims
    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)

    return image, label


def augment(image, label, apply_flip, x_bright, x_contrast, x_saturation, x_quality):
    """Data augmentation"""
    if x_bright !=0:
        image = tf.image.stateless_random_brightness(image=image, max_delta=x_bright, seed=[42,0])
    if x_contrast != 0:
        image = tf.image.stateless_random_contrast(image=image, lower=1-x_contrast, upper=1+x_contrast, seed=[42,0])
    if x_saturation != 0:
        image = tf.image.stateless_random_saturation(image=image, lower=1-x_saturation, upper=1+x_saturation, seed=[42,0])
    if x_quality != 100:
        image = tf.image.stateless_random_jpeg_quality(image=image, min_jpeg_quality=x_quality, max_jpeg_quality=100, seed=[42,0])
    if apply_flip == True:
        image = tf.image.stateless_random_flip_left_right(image=image, seed=[42,0])
        image = tf.image.stateless_random_flip_up_down(image=image,seed=[42,0])
    return image, label
